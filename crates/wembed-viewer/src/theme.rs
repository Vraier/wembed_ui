//! Base16 / base24 color themes for viewport background and categorical vertex palettes.
//!
//! Themes come from the [tinted-theming/schemes](https://github.com/tinted-theming/schemes)
//! ecosystem — YAML files with a fixed set of named colors (`base00`..`base0F` for base16,
//! extended to `base17` for base24). We use:
//!
//! - `base00` — viewport background (goes into the wgpu clear color).
//! - `base03` — edges under the `theme` edge-color source, unless the theme sets `edge`.
//! - `base09` — vertices in the `default` color mode.
//! - `base08`..`base0F` (base16) or `..base17` (base24) — categorical accent palette,
//!   mapped from `--categories` integers by `palette[category % palette.len()]`.
//! - `base04` — mid-gray, used for uncategorized vertices (`category = -1`).
//!
//! Loading is `--themes-dir` at CLI time (defaults to the repo's `themes/` folder next to
//! the shaders). Every `*.yaml` in that folder is parsed; parse failures log a warning
//! and skip the file — one bad theme doesn't kill the viewer. If no themes load at all,
//! the compile-in Tokyo Night fallback keeps the theme picker non-empty.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Fallback baked into the binary. Ensures the theme picker always has at least one
/// entry even if `--themes-dir` is misspecified or the folder is missing. The YAML lives
/// alongside the shaders in the repo so devs can swap it out.
const FALLBACK_THEME_YAML: &str = include_str!("../themes/catppuccin-mocha.yaml");

#[derive(Debug, Deserialize)]
struct RawTheme {
    /// `"base16"` or `"base24"` per the tinted-theming spec.
    system: String,
    name: String,
    palette: BTreeMap<String, String>,
    /// Optional `edge: "#rrggbb"` — the theme edge color (the `theme` edge-color source).
    /// A top-level key rather than a palette slot because it isn't part of the base16/24
    /// spec. Absent → `base03`.
    #[serde(default)]
    edge: Option<String>,
}

/// One loaded theme. Named + system + a fully-resolved RGBA8 table indexed by base slot
/// (0 = base00, 1 = base01, …).
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    /// File stem the theme was loaded from (empty for the compiled-in fallback).
    /// Matched by [`Theme::matches`] alongside `name`, so `--theme rose-pine`
    /// finds "Rosé Pine" via its filename even though the display name differs.
    pub slug: String,
    /// base16 vs base24 — kept for future callers that want to check accent capacity
    /// (accessible via [`Theme::accents`]`().len()` already, but the enum is more
    /// direct).
    #[allow(dead_code)]
    pub system: ThemeSystem,
    /// Length 16 for base16, 24 for base24. Each entry is `[R, G, B, A]` in sRGB byte
    /// order (little-endian pack matches the shader's `unpack4x8unorm`).
    colors: Vec<[u8; 4]>,
    /// The optional `edge` key, resolved. `None` → `base03`.
    edge: Option<[u8; 4]>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThemeSystem {
    Base16,
    Base24,
}

impl ThemeSystem {
    fn slot_count(self) -> usize {
        match self {
            ThemeSystem::Base16 => 16,
            ThemeSystem::Base24 => 24,
        }
    }
}

impl Theme {
    /// Parse a tinted-theming YAML source. Rejects malformed hex, missing slots, or
    /// unknown `system` values with a descriptive error — the caller decides whether to
    /// abort or just log-and-skip.
    pub fn from_yaml(source: &str) -> Result<Self> {
        let raw: RawTheme = serde_yaml_ng::from_str(source).context("YAML parse")?;
        let system = match raw.system.as_str() {
            "base16" => ThemeSystem::Base16,
            "base24" => ThemeSystem::Base24,
            other => anyhow::bail!("unknown theme system {other:?} (expected base16 or base24)"),
        };
        let slots = system.slot_count();
        let mut colors = vec![[0u8, 0, 0, 0xff]; slots];
        for (i, slot) in colors.iter_mut().enumerate() {
            let key = format!("base{i:02X}");
            let hex = raw
                .palette
                .get(&key)
                .with_context(|| format!("theme {:?}: missing key {key}", raw.name))?;
            *slot = parse_hex_rgba(hex)
                .with_context(|| format!("theme {:?}: bad hex for {key}: {hex:?}", raw.name))?;
        }
        let edge = raw
            .edge
            .as_deref()
            .map(parse_hex_rgba)
            .transpose()
            .with_context(|| format!("theme {:?}: bad hex for edge", raw.name))?;
        Ok(Self {
            name: raw.name,
            slug: String::new(),
            system,
            colors,
            edge,
        })
    }

    /// Case- and separator-insensitive match for `--theme`/`[theme].name`: query
    /// and candidates are lowercased and stripped of non-alphanumeric chars, so
    /// `tokyo-night-storm` matches "Tokyo Night Storm". Both the display name
    /// and the source file stem are candidates.
    pub fn matches(&self, query: &str) -> bool {
        let q = normalize(query);
        normalize(&self.name).contains(&q) || normalize(&self.slug).contains(&q)
    }

    /// Viewport background — `base00`.
    pub fn background(&self) -> [u8; 4] {
        self.colors[0]
    }

    /// Mid-gray neutral — `base04`. Used for `-1` (uncategorized) category entries.
    pub fn neutral(&self) -> [u8; 4] {
        self.colors[4]
    }

    /// Default foreground — `base05`. Used as the title text color when `[text].color`
    /// doesn't override it.
    pub fn foreground(&self) -> [u8; 4] {
        self.colors[5]
    }

    /// Edge color for the `theme` edge-color source — the optional `edge` key, else
    /// `base03` (base16's de-emphasized foreground: comments, invisibles).
    pub fn edge_color(&self) -> [u8; 4] {
        self.edge.unwrap_or(self.colors[3])
    }

    /// Vertex color of the `default` color mode — `base09`, the orange accent.
    pub fn vertex_color(&self) -> [u8; 4] {
        self.colors[9]
    }

    /// Categorical accent palette. Concatenates the primary accents (base08..base0F, 8
    /// slots) with the bright accents (base12..base17, 6 slots) for base24 themes —
    /// skipping base10/base11 which are additional *background* shades in the base24
    /// spec, not accents. Returns 8 colors for base16, 14 for base24.
    pub fn accents(&self) -> Vec<[u8; 4]> {
        // base08..base0F — primaries; present in every theme.
        let mut out: Vec<[u8; 4]> = self.colors[8..16].to_vec();
        // base12..base17 — bright variants; base24 only.
        if self.system == ThemeSystem::Base24 && self.colors.len() >= 24 {
            out.extend_from_slice(&self.colors[18..24]);
        }
        out
    }
}

/// Named collection of themes, loaded once at startup. `current` is the theme applied to
/// the viewport (bg + palette); the UI mutates it via the theme picker.
pub struct ThemeRegistry {
    themes: Vec<Theme>,
    current: usize,
}

impl ThemeRegistry {
    /// Build from a folder of `*.yaml` files. Files that fail to parse are logged and
    /// skipped. If none load successfully, falls back to the compiled-in default so the
    /// picker is never empty.
    ///
    /// The initial `current` is the first theme matching `preferred` (see
    /// [`Theme::matches`] — separator-insensitive substring against name or file
    /// stem); falls back to index 0 with a warning if none match.
    pub fn from_dir(dir: &Path, preferred: Option<&str>) -> Result<Self> {
        let mut themes: Vec<Theme> = Vec::new();
        match std::fs::read_dir(dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) != Some("yaml") {
                        continue;
                    }
                    match std::fs::read_to_string(&path)
                        .with_context(|| format!("read {}", path.display()))
                        .and_then(|src| Theme::from_yaml(&src))
                    {
                        Ok(mut t) => {
                            t.slug = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("")
                                .to_string();
                            themes.push(t);
                        }
                        Err(e) => eprintln!("warning: skipping theme {}: {e}", path.display()),
                    }
                }
            }
            Err(e) => eprintln!("warning: --themes-dir {} unreadable: {e}", dir.display()),
        }
        if themes.is_empty() {
            themes.push(Theme::from_yaml(FALLBACK_THEME_YAML).expect("built-in theme"));
        }
        // Sort by name for deterministic UI order.
        themes.sort_by_key(|t| t.name.to_lowercase());
        let current = preferred
            .and_then(|p| {
                let pos = themes.iter().position(|t| t.matches(p));
                if pos.is_none() {
                    eprintln!(
                        "warning: --theme {p:?} matches no loaded theme; using {:?}",
                        themes[0].name
                    );
                }
                pos
            })
            .unwrap_or(0);
        Ok(Self { themes, current })
    }

    pub fn current(&self) -> &Theme {
        &self.themes[self.current]
    }

    pub fn current_index(&self) -> usize {
        self.current
    }

    pub fn set_current(&mut self, index: usize) {
        if index < self.themes.len() {
            self.current = index;
        }
    }

    /// Iterator over (index, name) for the UI dropdown.
    pub fn names(&self) -> impl Iterator<Item = (usize, &str)> {
        self.themes
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.name.as_str()))
    }
}

/// Lowercase and drop non-alphanumeric chars, folding `-`/`_`/space away so
/// hyphenated CLI values match spaced display names.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Parse `"#RRGGBB"` or `"RRGGBB"` into `[R, G, B, 0xff]`. Rejects other lengths.
fn parse_hex_rgba(hex: &str) -> Result<[u8; 4]> {
    let s = hex.strip_prefix('#').unwrap_or(hex);
    if s.len() != 6 {
        anyhow::bail!("expected 6 hex chars, got {} ({s:?})", s.len());
    }
    let r = u8::from_str_radix(&s[0..2], 16).context("bad R")?;
    let g = u8::from_str_radix(&s[2..4], 16).context("bad G")?;
    let b = u8::from_str_radix(&s[4..6], 16).context("bad B")?;
    Ok([r, g, b, 0xff])
}

/// Pack `[R, G, B, A]` bytes into a little-endian `u32` matching the shader's
/// `unpack4x8unorm(colors[i])` — bits `0..8 = R`, `8..16 = G`, `16..24 = B`, `24..32 = A`.
pub fn pack_rgba(rgba: [u8; 4]) -> u32 {
    u32::from_le_bytes(rgba)
}

/// Normalize `[R, G, B, A]` bytes to `[0, 1]` floats, matching `unpack4x8unorm` (a straight
/// `byte / 255`, still sRGB). The lines shader decodes the `edge_theme_color` uniform exactly
/// like the unpacked vertex colors, so the theme's `edge` key uses the same authoring
/// convention as a `--colors` hex file.
pub fn rgba_to_unorm(rgba: [u8; 4]) -> [f32; 4] {
    [
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
        rgba[3] as f32 / 255.0,
    ]
}

/// sRGB `[R, G, B]` bytes → wgpu linear `Color`. wgpu treats clear-color values as
/// linear when writing to an sRGB target, so a "black" clear needs to be linear zero and
/// a hex `#1A1B26` needs sRGB-to-linear conversion first.
pub fn wgpu_clear_color(rgba: [u8; 4]) -> wgpu::Color {
    wgpu::Color {
        r: srgb_to_linear(rgba[0]) as f64,
        g: srgb_to_linear(rgba[1]) as f64,
        b: srgb_to_linear(rgba[2]) as f64,
        a: rgba[3] as f64 / 255.0,
    }
}

/// sRGB byte → linear float (standard sRGB EOTF).
pub(crate) fn srgb_to_linear(byte: u8) -> f32 {
    let s = byte as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear float → sRGB byte. Inverse of [`srgb_to_linear`], clamped.
pub(crate) fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a base16 YAML with distinct colors per slot.
    fn base16_yaml() -> String {
        let mut s = String::from("system: \"base16\"\nname: \"Test16\"\npalette:\n");
        for i in 0..16u32 {
            let byte = i as u8;
            s.push_str(&format!(
                "  base{i:02X}: \"#{byte:02x}{byte:02x}{byte:02x}\"\n"
            ));
        }
        s
    }

    /// Build a base24 YAML with distinct colors per slot.
    fn base24_yaml() -> String {
        let mut s = String::from("system: \"base24\"\nname: \"Test24\"\npalette:\n");
        for i in 0..24u32 {
            let byte = i as u8;
            s.push_str(&format!(
                "  base{i:02X}: \"#{byte:02x}{byte:02x}{byte:02x}\"\n"
            ));
        }
        s
    }

    // ---- Theme::from_yaml ----

    #[test]
    fn base16_has_16_colors_bg_base00_neutral_base04() {
        let theme = Theme::from_yaml(&base16_yaml()).unwrap();
        assert_eq!(theme.system, ThemeSystem::Base16);
        // base00 → background (slot 0 = [0,0,0,ff]).
        assert_eq!(theme.background(), [0, 0, 0, 0xff]);
        // base04 → neutral (slot 4 = [4,4,4,ff]).
        assert_eq!(theme.neutral(), [4, 4, 4, 0xff]);
    }

    #[test]
    fn base24_accents_have_14_entries() {
        let theme = Theme::from_yaml(&base24_yaml()).unwrap();
        let accents = theme.accents();
        assert_eq!(accents.len(), 14);
        // First 8 are base08..base0F.
        assert_eq!(accents[0], [8, 8, 8, 0xff]);
        assert_eq!(accents[7], [0x0f, 0x0f, 0x0f, 0xff]);
        // Next 6 are base12..base17 (base10/base11 skipped).
        assert_eq!(accents[8], [0x12, 0x12, 0x12, 0xff]);
        assert_eq!(accents[13], [0x17, 0x17, 0x17, 0xff]);
    }

    #[test]
    fn base16_accents_have_8_entries() {
        let theme = Theme::from_yaml(&base16_yaml()).unwrap();
        let accents = theme.accents();
        assert_eq!(accents.len(), 8);
        assert_eq!(accents[0], [8, 8, 8, 0xff]);
        assert_eq!(accents[7], [0x0f, 0x0f, 0x0f, 0xff]);
    }

    #[test]
    fn from_yaml_missing_key_errors_naming_key() {
        // Drop base05 from an otherwise-valid base16.
        let mut s = String::from("system: \"base16\"\nname: \"Broken\"\npalette:\n");
        for i in 0..16u32 {
            if i == 5 {
                continue;
            }
            s.push_str(&format!("  base{i:02X}: \"#000000\"\n"));
        }
        let err = Theme::from_yaml(&s).unwrap_err().to_string();
        assert!(
            err.contains("base05"),
            "error should name missing key: {err}"
        );
    }

    #[test]
    fn from_yaml_bad_hex_errors() {
        let mut s = String::from("system: \"base16\"\nname: \"BadHex\"\npalette:\n");
        for i in 0..16u32 {
            let hex = if i == 3 { "#zzzzzz" } else { "#000000" };
            s.push_str(&format!("  base{i:02X}: \"{hex}\"\n"));
        }
        let err = Theme::from_yaml(&s).unwrap_err().to_string();
        assert!(
            err.contains("base03"),
            "error should name the bad slot: {err}"
        );
    }

    #[test]
    fn from_yaml_unknown_system_errors() {
        let s = "system: \"base99\"\nname: \"Nope\"\npalette: {}\n";
        let err = Theme::from_yaml(s).unwrap_err().to_string();
        assert!(
            err.contains("unknown theme system"),
            "unexpected error: {err}"
        );
    }

    // ---- Theme::matches ----

    #[test]
    fn matches_ignores_case_and_separators_and_checks_slug() {
        let mut theme =
            Theme::from_yaml(&base16_yaml().replace("\"Test16\"", "\"Tokyo Night Storm\""))
                .unwrap();
        assert!(theme.matches("tokyo-night-storm"), "hyphens vs spaces");
        assert!(theme.matches("Tokyo Night"), "plain substring still works");
        assert!(!theme.matches("nord"));
        // Slug (file stem) is a candidate too — covers names the query can't
        // reproduce, like "Rosé Pine" from `rose-pine.yaml`.
        theme.name = "Rosé Pine".into();
        theme.slug = "rose-pine".into();
        assert!(theme.matches("rose-pine"));
    }

    // ---- parse_hex_rgba / pack_rgba / wgpu_clear_color ----

    #[test]
    fn parse_hex_rgba_with_and_without_hash() {
        assert_eq!(parse_hex_rgba("#1a2b3c").unwrap(), [0x1a, 0x2b, 0x3c, 0xff]);
        assert_eq!(parse_hex_rgba("1A2B3C").unwrap(), [0x1a, 0x2b, 0x3c, 0xff]);
    }

    #[test]
    fn parse_hex_rgba_wrong_length_errors() {
        assert!(parse_hex_rgba("#12345").is_err());
        assert!(parse_hex_rgba("1234567").is_err());
    }

    #[test]
    fn pack_rgba_matches_unpack4x8unorm_byte_order() {
        // WGSL unpack4x8unorm reads byte 0 as .x (R). So packed bits 0..8 must be R.
        let packed = pack_rgba([0x11, 0x22, 0x33, 0x44]);
        assert_eq!(packed & 0xff, 0x11, "R in low byte");
        assert_eq!((packed >> 8) & 0xff, 0x22, "G");
        assert_eq!((packed >> 16) & 0xff, 0x33, "B");
        assert_eq!((packed >> 24) & 0xff, 0x44, "A");
    }

    #[test]
    fn srgb_to_linear_endpoints_and_midpoint() {
        let eps = 1e-6_f64;
        let c = wgpu_clear_color([0, 255, 128, 255]);
        assert!((c.r - 0.0).abs() < eps, "0 → 0.0, got {}", c.r);
        assert!((c.g - 1.0).abs() < eps, "255 → 1.0, got {}", c.g);
        // Spot-check a mid value: srgb 128/255 ≈ 0.50196 → linear ≈ 0.21586.
        assert!((c.b - 0.2158605).abs() < 1e-4, "128 midpoint, got {}", c.b);
        // Alpha is a straight /255.
        assert!((c.a - 1.0).abs() < eps, "alpha 255 → 1.0, got {}", c.a);
    }
}
