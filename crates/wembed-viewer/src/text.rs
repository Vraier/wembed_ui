//! Title text: CPU rasterization + anchor math.
//!
//! The design: text is rasterized ONCE on the CPU into an
//! RGBA bitmap; a textured quad in [`crate::scene_renderer`] composites that bitmap in
//! BOTH the window pass and the offscreen export pass — same blend math, pixel-identical
//! preview by construction. This module owns everything CPU-side: font loading, glyph
//! layout + rasterization, and the anchor→pixel-rect placement shared by every caller.
//!
//! Determinism: the bundled font is compiled in (`include_bytes!`), no system font
//! discovery — identical PNGs on every machine. `[text].font` overrides per config.
//!
//! Sizes are specified in px at the 1080p reference height and scaled by the caller
//! (same rule as the point-size pixel clamps), re-rasterizing rather than stretching so
//! 4K exports stay crisp.

use std::path::Path;

use ab_glyph::{Font, FontArc, Glyph, ScaleFont};
use anyhow::{Context, Result};

/// Bundled fallback typeface — JetBrains Mono Regular (OFL, license alongside the file).
/// Matches the reference-height convention: a `size` of 28 reads like 28px text on a
/// 1080p window.
const BUNDLED_FONT: &[u8] = include_bytes!("../fonts/JetBrainsMono-Regular.ttf");

/// Corner the title block is anchored to, with a margin. Serde names are kebab-case to
/// read naturally in TOML (`anchor = "top-left"`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Anchor {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// A rasterized text block: straight-alpha RGBA8, tightly cropped.
pub struct TextBitmap {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Parse `#rrggbb` / `#rrggbbaa` (leading `#` optional, case-insensitive) into RGBA
/// bytes. Alpha defaults to opaque. Used for `[text].color`, validated at resolve time
/// so a typo fails before the graph load.
pub fn parse_color(s: &str) -> Result<[u8; 4]> {
    let hex = s.strip_prefix('#').unwrap_or(s);
    let parse2 = |i: usize| {
        u8::from_str_radix(&hex[i..i + 2], 16)
            .with_context(|| format!("bad hex color {s:?} at digit {i}"))
    };
    match hex.len() {
        6 => Ok([parse2(0)?, parse2(2)?, parse2(4)?, 0xff]),
        8 => Ok([parse2(0)?, parse2(2)?, parse2(4)?, parse2(6)?]),
        n => anyhow::bail!("bad hex color {s:?}: expected 6 or 8 digits, got {n}"),
    }
}

/// Load the title font: `path` if configured (fail fast on a bad file), else the
/// compiled-in fallback.
pub fn load_font(path: Option<&Path>) -> Result<FontArc> {
    match path {
        Some(p) => {
            let bytes =
                std::fs::read(p).with_context(|| format!("read font file {}", p.display()))?;
            FontArc::try_from_vec(bytes).with_context(|| format!("parse font {}", p.display()))
        }
        None => FontArc::try_from_slice(BUNDLED_FONT).context("parse bundled font"),
    }
}

/// Rasterize `text` at `px` pixels tall (per line) in `color` (sRGB bytes; alpha is the
/// text opacity). Supports `\n` for multi-line; lines are left-aligned. Returns `None`
/// for effectively-empty input (nothing to draw — callers skip the quad entirely).
///
/// Output is straight (non-premultiplied) alpha, matching the pipelines'
/// `ALPHA_BLENDING` state; the sRGB texture format handles color-space conversion when
/// the GPU samples it.
pub fn rasterize(font: &FontArc, text: &str, px: f32, color: [u8; 4]) -> Option<TextBitmap> {
    if text.trim().is_empty() || px < 1.0 {
        return None;
    }
    let scaled = font.as_scaled(ab_glyph::PxScale::from(px));
    let line_height = (scaled.ascent() - scaled.descent() + scaled.line_gap()).ceil();
    let ascent = scaled.ascent();

    // Layout: position every glyph (with kerning), tracking the tight bounding box of
    // the outlined glyphs. Whitespace-only lines still advance the cursor.
    let mut glyphs: Vec<Glyph> = Vec::new();
    let mut max_line_width = 0.0f32;
    let mut baseline_y = ascent;
    for line in text.split('\n') {
        let mut x = 0.0f32;
        let mut prev: Option<ab_glyph::GlyphId> = None;
        for ch in line.chars() {
            let id = scaled.glyph_id(ch);
            if let Some(p) = prev {
                x += scaled.kern(p, id);
            }
            glyphs.push(id.with_scale_and_position(px, ab_glyph::point(x, baseline_y)));
            x += scaled.h_advance(id);
            prev = Some(id);
        }
        max_line_width = max_line_width.max(x);
        baseline_y += line_height;
    }
    if glyphs.is_empty() {
        return None;
    }

    let width = max_line_width.ceil() as u32;
    // `baseline_y` advanced one line past the last baseline; block height = last
    // baseline + descent depth (ab_glyph's descent() is negative).
    let last_baseline = baseline_y - line_height;
    let height = (last_baseline - scaled.descent()).ceil() as u32;
    if width == 0 || height == 0 {
        return None;
    }

    // Coverage accumulation, then colorize. Glyphs can overlap (kerning, italics) —
    // take the max coverage so overlaps don't darken.
    let mut coverage = vec![0.0f32; (width * height) as usize];
    for glyph in glyphs {
        let Some(outlined) = font.outline_glyph(glyph) else {
            continue; // whitespace / glyphless
        };
        let bounds = outlined.px_bounds();
        outlined.draw(|gx, gy, c| {
            let x = bounds.min.x as i64 + gx as i64;
            let y = bounds.min.y as i64 + gy as i64;
            if x >= 0 && (x as u32) < width && y >= 0 && (y as u32) < height {
                let idx = (y as u32 * width + x as u32) as usize;
                coverage[idx] = coverage[idx].max(c);
            }
        });
    }

    let mut rgba = Vec::with_capacity(coverage.len() * 4);
    let alpha_scale = color[3] as f32 / 255.0;
    for c in coverage {
        rgba.extend_from_slice(&[
            color[0],
            color[1],
            color[2],
            (c * alpha_scale * 255.0).round() as u8,
        ]);
    }
    Some(TextBitmap {
        rgba,
        width,
        height,
    })
}

/// Rasterize a legend block: one row per `(label, swatch_color)` entry — a filled color
/// square sitting on the text baseline, a gap, then the label in `text_color`. Same
/// output contract as [`rasterize`] (straight-alpha RGBA8, tightly cropped, `None` when
/// there is nothing to draw), so it feeds the same textured-quad path.
///
/// All rows share the font's baseline grid — per-row tight-cropping would make baselines
/// jitter between short and tall labels.
pub fn rasterize_legend(
    font: &FontArc,
    rows: &[(String, [u8; 4])],
    px: f32,
    text_color: [u8; 4],
) -> Option<TextBitmap> {
    if rows.is_empty() || px < 1.0 {
        return None;
    }
    let scaled = font.as_scaled(ab_glyph::PxScale::from(px));
    // A little extra leading over plain multi-line text — legend rows are independent
    // items, not a paragraph, and read better with air between the swatches.
    let row_pitch = ((scaled.ascent() - scaled.descent() + scaled.line_gap()) * 1.15).ceil();
    let ascent = scaled.ascent();
    // Swatch side ≈ cap height, so square and uppercase text read as the same size.
    let square = (px * 0.7).round().max(1.0);
    let text_x0 = square + (px * 0.4).round();

    // Glyph layout on the shared baseline grid, tracking the widest row.
    let mut glyphs: Vec<Glyph> = Vec::new();
    let mut max_width = text_x0;
    let mut baseline_y = ascent;
    for (label, _) in rows {
        let mut x = text_x0;
        let mut prev: Option<ab_glyph::GlyphId> = None;
        for ch in label.chars() {
            let id = scaled.glyph_id(ch);
            if let Some(p) = prev {
                x += scaled.kern(p, id);
            }
            glyphs.push(id.with_scale_and_position(px, ab_glyph::point(x, baseline_y)));
            x += scaled.h_advance(id);
            prev = Some(id);
        }
        max_width = max_width.max(x);
        baseline_y += row_pitch;
    }

    let width = max_width.ceil() as u32;
    let last_baseline = baseline_y - row_pitch;
    let height = (last_baseline - scaled.descent()).ceil() as u32;
    if width == 0 || height == 0 {
        return None;
    }

    // Text coverage → colorize, exactly like `rasterize`.
    let mut coverage = vec![0.0f32; (width * height) as usize];
    for glyph in glyphs {
        let Some(outlined) = font.outline_glyph(glyph) else {
            continue;
        };
        let bounds = outlined.px_bounds();
        outlined.draw(|gx, gy, c| {
            let x = bounds.min.x as i64 + gx as i64;
            let y = bounds.min.y as i64 + gy as i64;
            if x >= 0 && (x as u32) < width && y >= 0 && (y as u32) < height {
                let idx = (y as u32 * width + x as u32) as usize;
                coverage[idx] = coverage[idx].max(c);
            }
        });
    }
    let mut rgba = Vec::with_capacity(coverage.len() * 4);
    let alpha_scale = text_color[3] as f32 / 255.0;
    for c in coverage {
        rgba.extend_from_slice(&[
            text_color[0],
            text_color[1],
            text_color[2],
            (c * alpha_scale * 255.0).round() as u8,
        ]);
    }

    // Swatches on top: a filled square per row, bottom edge on the row's baseline.
    // Labels start at `text_x0 > square`, so squares never overwrite text.
    for (i, (_, color)) in rows.iter().enumerate() {
        let baseline = (ascent + i as f32 * row_pitch).round() as i64;
        let top = (baseline - square as i64).max(0);
        for y in top..baseline.min(height as i64) {
            for x in 0..(square as i64).min(width as i64) {
                let idx = ((y as u32 * width + x as u32) * 4) as usize;
                rgba[idx..idx + 4].copy_from_slice(color);
            }
        }
    }

    Some(TextBitmap {
        rgba,
        width,
        height,
    })
}

/// Where a `bitmap_w × bitmap_h` block lands inside an anchor rect (all in pixels of the
/// same target). `rect` is `(x, y, w, h)` — the full target for exports, the window or
/// the export-frame overlay for the interactive preview. Margin scales with the text
/// size so the inset reads consistently across resolutions. Returns `(x, y)` of the
/// block's top-left corner.
pub fn anchor_position(
    anchor: Anchor,
    rect: (f32, f32, f32, f32),
    bitmap_w: f32,
    bitmap_h: f32,
    margin_x: f32,
    margin_y: f32,
) -> (f32, f32) {
    let (rx, ry, rw, rh) = rect;
    let x = match anchor {
        Anchor::TopLeft | Anchor::BottomLeft => rx + margin_x,
        Anchor::TopRight | Anchor::BottomRight => rx + rw - margin_x - bitmap_w,
    };
    let y = match anchor {
        Anchor::TopLeft | Anchor::TopRight => ry + margin_y,
        Anchor::BottomLeft | Anchor::BottomRight => ry + rh - margin_y - bitmap_h,
    };
    (x, y)
}

/// Pixel rect → NDC rect (`[x0, y0, x1, y1]`, y up) for the text quad uniform.
pub fn pixel_rect_to_ndc(x: f32, y: f32, w: f32, h: f32, target_w: f32, target_h: f32) -> [f32; 4] {
    let x0 = 2.0 * x / target_w - 1.0;
    let x1 = 2.0 * (x + w) / target_w - 1.0;
    // Pixel y grows down, NDC y grows up.
    let y0 = 1.0 - 2.0 * (y + h) / target_h;
    let y1 = 1.0 - 2.0 * y / target_h;
    [x0, y0, x1, y1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn font() -> FontArc {
        load_font(None).expect("bundled font parses")
    }

    // The bundled font must parse and produce non-empty bitmaps — guards against a bad
    // font file landing in the repo (include_bytes! can't check).
    #[test]
    fn bundled_font_rasterizes_nonempty() {
        let bmp = rasterize(&font(), "Karate", 28.0, [255, 255, 255, 255]).expect("some bitmap");
        assert!(bmp.width > 0 && bmp.height > 0);
        // Some pixel must actually be inked.
        assert!(bmp.rgba.chunks_exact(4).any(|p| p[3] > 0));
    }

    // Empty / whitespace-only titles must yield None — the callers use this to skip the
    // quad; a Some here would draw a stray dot or panic on a zero-sized texture.
    #[test]
    fn empty_and_whitespace_yield_none() {
        assert!(rasterize(&font(), "", 28.0, [255; 4]).is_none());
        assert!(rasterize(&font(), "   \n  ", 28.0, [255; 4]).is_none());
    }

    // Multi-line must be taller than single-line — catches a layout bug where `\n`
    // advances x instead of the baseline.
    #[test]
    fn multiline_grows_height_not_width() {
        let one = rasterize(&font(), "ab", 28.0, [255; 4]).unwrap();
        let two = rasterize(&font(), "ab\nab", 28.0, [255; 4]).unwrap();
        assert!(two.height > one.height);
        assert_eq!(two.width, one.width);
    }

    // Anchor math: the four corners must each respect margin from their own edges —
    // sign errors here put the title off-canvas.
    #[test]
    fn anchor_positions_respect_margins() {
        let rect = (0.0, 0.0, 100.0, 50.0);
        let (bw, bh, m) = (20.0, 10.0, 5.0);
        assert_eq!(
            anchor_position(Anchor::TopLeft, rect, bw, bh, m, m),
            (5.0, 5.0)
        );
        assert_eq!(
            anchor_position(Anchor::TopRight, rect, bw, bh, m, m),
            (75.0, 5.0)
        );
        assert_eq!(
            anchor_position(Anchor::BottomLeft, rect, bw, bh, m, m),
            (5.0, 35.0)
        );
        assert_eq!(
            anchor_position(Anchor::BottomRight, rect, bw, bh, m, m),
            (75.0, 35.0)
        );
    }

    // NDC conversion: full-target rect maps to clip corners; y must flip — a sign error
    // here renders the title mirrored or off-screen.
    #[test]
    fn ndc_rect_flips_y() {
        let ndc = pixel_rect_to_ndc(0.0, 0.0, 100.0, 50.0, 100.0, 50.0);
        assert_eq!(ndc, [-1.0, -1.0, 1.0, 1.0]);
        // A rect at the TOP of the target must land at ndc y ≈ +1.
        let top = pixel_rect_to_ndc(0.0, 0.0, 10.0, 10.0, 100.0, 100.0);
        assert!(top[3] == 1.0 && top[1] > 0.0);
    }
}
