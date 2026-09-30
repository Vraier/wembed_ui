//! Framing + point-sizing helpers shared by the interactive App and the headless batch
//! path. Pure math over positions and sizes — no wgpu, no winit — so both callers (and
//! unit tests) can use it without a window.

use glam::Mat4;

use crate::camera::Camera;

/// Quantile-based framing bounds. Positions outside [Q_LO, Q_HI] per axis are candidates
/// for outlier clipping; whether they're actually clipped depends on Q_BUFFER (below).
pub const Q_LO: f32 = 0.005;
pub const Q_HI: f32 = 1.0 - Q_LO;
/// Fraction of the quantile span that framing may extend beyond the quantile bounds to
/// include the true extremes. If the extremes fall inside the buffer, the whole graph fits;
/// otherwise outliers are clipped at the buffer edge.
pub const Q_BUFFER: f32 = 0.1;

/// Viewer-side sizing knobs (as opposed to embedder-side knobs which live in
/// `SpawnConfig`).
#[derive(Clone, Debug)]
pub struct SizingConfig {
    /// Point radius as a fraction of estimated nearest-neighbor spacing, before pixel
    /// clamping. Diameter ≈ 2 * factor * spacing — a factor of 0.35 leaves ~30% gap in
    /// dense areas.
    pub density_factor: f32,
    /// Visibility floor in screen pixels. Prevents dots from disappearing at high N.
    pub min_pixels: f32,
    /// Ceiling in screen pixels. Prevents dots from dominating the frame at low N.
    pub max_pixels: f32,
}

/// Hybrid clamped point radius in world units. Density-driven target, capped and floored by
/// pixel-space bounds — all evaluated at the camera's fitted framing, so the radius stays put
/// in world units while zooming and dots grow like a crop of the unzoomed picture.
pub fn point_radius(
    cam: &Camera,
    num_vertices: usize,
    width: u32,
    height: u32,
    sizing: &SizingConfig,
) -> f32 {
    let n = num_vertices.max(1) as f32;
    let density_target = sizing.density_factor * 2.0 * cam.fit_half() / n.sqrt();
    let wpp = cam.world_per_pixel(width, height) * cam.zoom();
    let min_world = sizing.min_pixels * wpp;
    let max_world = sizing.max_pixels * wpp;
    // Order matters: cap the density-derived target at MAX first, then floor at MIN — so a
    // huge-N graph doesn't produce a density below MIN that then gets pushed past MAX.
    density_target.min(max_world).max(min_world)
}

/// Edge stroke width in device pixels before the `edge_width_scale` multiplier: 1px at
/// [`REFERENCE_HEIGHT`], scaled with the render height (so exports keep the window's
/// proportions) and with the zoom (so zooming reads as a crop, like the dots). The shader
/// floors the final width at 1px.
pub fn line_width_px(cam: &Camera, height: u32) -> f32 {
    height as f32 / REFERENCE_HEIGHT as f32 * cam.zoom()
}

/// Everything the scene shaders need to draw `cam` into a `width × height` target. Built in
/// one place so the window, the interactive export and headless can't drift apart.
#[derive(Copy, Clone, Debug)]
pub struct View {
    pub view_proj: Mat4,
    /// World units.
    pub point_radius: f32,
    pub line_width_px: f32,
    pub width: u32,
    pub height: u32,
}

impl View {
    /// `sizing` is the `REFERENCE_HEIGHT`-relative config; scaled to `height` here.
    pub fn new(
        cam: &Camera,
        num_vertices: usize,
        width: u32,
        height: u32,
        sizing: &SizingConfig,
    ) -> Self {
        let sizing = scaled_sizing(sizing, height, REFERENCE_HEIGHT);
        Self {
            view_proj: cam.view_proj(width, height),
            point_radius: point_radius(cam, num_vertices, width, height, &sizing),
            line_width_px: line_width_px(cam, height),
            width,
            height,
        }
    }
}

/// Canonical reference height for the pixel clamps. `--point-{min,max}-pixels` are defined
/// as "pixels at 1080p" everywhere — the live window, the interactive export, and headless
/// all scale them by `actual_height / REFERENCE_HEIGHT` via `scaled_sizing`, so the same
/// config produces the same visual proportion at any window size or export resolution.
pub const REFERENCE_HEIGHT: u32 = 1080;

/// Build a `SizingConfig` for a render at `render_height` that visually matches the pixel
/// clamps' `REFERENCE_HEIGHT`-relative definition. Only the pixel clamps scale — density
/// factor is unitless. Every caller (live window, interactive export, headless) passes the
/// same `REFERENCE_HEIGHT` so a config's dot sizes are resolution-independent.
pub fn scaled_sizing(
    sizing: &SizingConfig,
    render_height: u32,
    reference_height: u32,
) -> SizingConfig {
    let scale = (render_height as f32) / (reference_height.max(1) as f32);
    SizingConfig {
        density_factor: sizing.density_factor,
        min_pixels: sizing.min_pixels * scale,
        max_pixels: sizing.max_pixels * scale,
    }
}

/// Brute-force autorotate: try every whole degree in `[0, 180)` (rotations beyond that
/// duplicate up to a flip that doesn't matter for framing aspect), pick the angle whose
/// rotated bounding-box aspect best matches `target_aspect`. Uses raw min/max bounds
/// (not quantile) — fast and good enough for framing.
///
/// Comparison metric is `|log(bbox_aspect / target_aspect)|` — symmetric on the "too
/// wide" vs "too tall" axis so a 2:1 target beats a 4:1 result the same as a 1:2 does.
pub fn best_rotation_deg_for_aspect(positions: &[f32], target_aspect: f32) -> f32 {
    debug_assert!(positions.len().is_multiple_of(2));
    if positions.is_empty() || !target_aspect.is_finite() || target_aspect <= 0.0 {
        return 0.0;
    }
    let log_target = target_aspect.ln();
    let mut best_deg = 0.0f32;
    let mut best_err = f32::INFINITY;
    for deg in 0..180 {
        let theta = (deg as f32).to_radians();
        let (c, s) = (theta.cos(), theta.sin());
        let (mut min_rx, mut max_rx) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_ry, mut max_ry) = (f32::INFINITY, f32::NEG_INFINITY);
        for xy in positions.chunks_exact(2) {
            let (x, y) = (xy[0], xy[1]);
            let rx = c * x - s * y;
            let ry = s * x + c * y;
            min_rx = min_rx.min(rx);
            max_rx = max_rx.max(rx);
            min_ry = min_ry.min(ry);
            max_ry = max_ry.max(ry);
        }
        let w = (max_rx - min_rx).max(1e-6);
        let h = (max_ry - min_ry).max(1e-6);
        let err = ((w / h).ln() - log_target).abs();
        if err < best_err {
            best_err = err;
            best_deg = deg as f32;
        }
    }
    best_deg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::{Q_BUFFER, Q_HI, Q_LO};

    const EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    /// Build a camera framing a horizontal segment of world-width `d`. With a zero-height
    /// span, `fit_to_quantile_flat` sets `half = d * 0.5 * 1.2`.
    fn camera_with_diameter(d: f32) -> Camera {
        let positions = vec![-d / 2.0, 0.0, d / 2.0, 0.0];
        Camera::fit_to_quantile_flat(&positions, Q_LO, Q_HI, Q_BUFFER)
    }

    fn sizing(density: f32, min_px: f32, max_px: f32) -> SizingConfig {
        SizingConfig {
            density_factor: density,
            min_pixels: min_px,
            max_pixels: max_px,
        }
    }

    // ---- point_radius ----

    #[test]
    fn point_radius_density_target_within_bounds_unchanged() {
        let cam = camera_with_diameter(100.0); // half = 60
        let (w, h) = (1000u32, 1000u32);
        // density_target = factor * diameter / sqrt(N). Pick N=4 → sqrt=2.
        let cfg = sizing(0.1, 0.1, 100.0);
        let density_target = 0.1 * 2.0 * cam.half() / 2.0;
        let wpp = cam.world_per_pixel(w, h);
        let min_world = cfg.min_pixels * wpp;
        let max_world = cfg.max_pixels * wpp;
        // Confirm the target actually sits inside the pixel bounds for this fixture.
        assert!(density_target > min_world && density_target < max_world);
        let r = point_radius(&cam, 4, w, h, &cfg);
        assert!(close(r, density_target), "got {r}, want {density_target}");
    }

    #[test]
    fn point_radius_huge_n_clamped_up_to_min() {
        let cam = camera_with_diameter(100.0);
        let (w, h) = (1000u32, 1000u32);
        let cfg = sizing(0.1, 5.0, 100.0);
        // Huge N pushes density target near zero → floored at min_pixels * wpp.
        let wpp = cam.world_per_pixel(w, h);
        let r = point_radius(&cam, 1_000_000, w, h, &cfg);
        assert!(close(r, cfg.min_pixels * wpp), "got {r}");
    }

    #[test]
    fn point_radius_tiny_n_clamped_down_to_max() {
        let cam = camera_with_diameter(100.0);
        let (w, h) = (1000u32, 1000u32);
        // Big density factor + N=1 → target huge → capped at max_pixels * wpp.
        let cfg = sizing(10.0, 1.0, 3.0);
        let wpp = cam.world_per_pixel(w, h);
        let r = point_radius(&cam, 1, w, h, &cfg);
        assert!(close(r, cfg.max_pixels * wpp), "got {r}");
    }

    #[test]
    fn point_radius_min_greater_than_max_min_wins() {
        let cam = camera_with_diameter(100.0);
        let (w, h) = (1000u32, 1000u32);
        // Documented order: cap at MAX first, then floor at MIN. With min > max, the
        // final `.max(min_world)` wins regardless of the density target.
        let cfg = sizing(1.0, 20.0, 5.0);
        let wpp = cam.world_per_pixel(w, h);
        let r = point_radius(&cam, 100, w, h, &cfg);
        assert!(close(r, cfg.min_pixels * wpp), "got {r}");
    }

    #[test]
    fn point_radius_zero_vertices_no_panic() {
        let cam = camera_with_diameter(100.0);
        let cfg = sizing(0.5, 3.0, 40.0);
        // num_vertices = 0 uses the max(1) path — must not divide-by-zero / panic.
        let r = point_radius(&cam, 0, 800, 600, &cfg);
        assert!(r.is_finite() && r > 0.0, "got {r}");
    }

    #[test]
    fn zoom_scales_dots_and_edges_like_a_crop() {
        // Zooming must magnify, not re-lay-out: the dot keeps its world radius (so its
        // pixel size grows with the zoom, clamps included) and the edge width grows by the
        // same factor. Sizing against the zoomed camera instead keeps dots at constant
        // pixel size while they spread apart.
        let (w, h) = (1000u32, 1000u32);
        let cfg = sizing(10.0, 1.0, 3.0); // capped at max_pixels: the clamp must scale too
        let mut cam = camera_with_diameter(100.0);
        let r0 = point_radius(&cam, 1, w, h, &cfg);
        let px0 = r0 / cam.world_per_pixel(w, h);
        cam.zoom_at_pixel(0.25, 500.0, 500.0, w, h); // 4× in
        let r1 = point_radius(&cam, 1, w, h, &cfg);
        assert!(close(r1, r0), "world radius changed: {r0} → {r1}");
        assert!(close(r1 / cam.world_per_pixel(w, h), 4.0 * px0));
        assert!(close(line_width_px(&cam, 1080), 4.0));
        // a `[camera]` restore (center + half on top of the fit) must reproduce the sizes, or
        // a saved zoomed view renders differently headless than in the window
        let restored = camera_with_diameter(100.0).with_view(cam.center(), cam.half());
        assert!(close(point_radius(&restored, 1, w, h, &cfg), r1));
    }

    // ---- scaled_sizing ----

    #[test]
    fn scaled_sizing_scales_pixels_by_height_ratio() {
        let base = sizing(0.42, 3.0, 40.0);
        let scaled = scaled_sizing(&base, 2160, 1080); // scale = 2.0
        assert!(close(scaled.density_factor, 0.42), "density untouched");
        assert!(close(scaled.min_pixels, 6.0), "min scaled");
        assert!(close(scaled.max_pixels, 80.0), "max scaled");
    }

    #[test]
    fn scaled_sizing_zero_reference_no_divide_by_zero() {
        let base = sizing(0.42, 3.0, 40.0);
        // reference_height = 0 uses max(1) → scale = render_height / 1.
        let scaled = scaled_sizing(&base, 100, 0);
        assert!(scaled.min_pixels.is_finite() && scaled.max_pixels.is_finite());
        assert!(close(scaled.min_pixels, 300.0), "got {}", scaled.min_pixels);
    }

    // ---- best_rotation_deg_for_aspect ----

    #[test]
    fn best_rotation_early_return_guards_are_zero() {
        // Empty positions and any non-finite / non-positive target all hit the `return 0.0`
        // guard before the search loop.
        assert!(close(best_rotation_deg_for_aspect(&[], 2.0), 0.0));
        let pts = vec![0.0, 0.0, 4.0, 1.0];
        assert!(close(best_rotation_deg_for_aspect(&pts, f32::NAN), 0.0));
        assert!(close(
            best_rotation_deg_for_aspect(&pts, f32::INFINITY),
            0.0
        ));
        assert!(close(best_rotation_deg_for_aspect(&pts, 0.0), 0.0));
        assert!(close(best_rotation_deg_for_aspect(&pts, -1.0), 0.0));
    }

    #[test]
    fn best_rotation_axis_aligned_wide_cloud() {
        // A 4:1 (wide) axis-aligned cloud. A wide target → ~0°, a tall target → ~90°.
        let pts = vec![-2.0, -0.5, 2.0, -0.5, 2.0, 0.5, -2.0, 0.5];
        let wide = best_rotation_deg_for_aspect(&pts, 4.0);
        assert!(wide <= 1.0 || wide >= 179.0, "wide target got {wide}");
        let tall = best_rotation_deg_for_aspect(&pts, 0.25);
        assert!((tall - 90.0).abs() <= 1.0, "tall target got {tall}");
    }

    #[test]
    fn best_rotation_diagonal_cloud_matches_diagonal_target() {
        // A wide cloud rotated 45°. Rotating it back by 45° recovers the wide bbox, so the
        // best angle for a wide (4:1) target is ~45° (or its 180-complement ~135°).
        let base = [(-2.0_f32, -0.5), (2.0, -0.5), (2.0, 0.5), (-2.0, 0.5)];
        let theta = 45.0_f32.to_radians();
        let (c, s) = (theta.cos(), theta.sin());
        let mut pts = Vec::new();
        for (x, y) in base {
            pts.push(c * x - s * y);
            pts.push(s * x + c * y);
        }
        let deg = best_rotation_deg_for_aspect(&pts, 4.0);
        let near_45 = (deg - 45.0).abs() <= 1.0;
        let near_135 = (deg - 135.0).abs() <= 1.0;
        assert!(near_45 || near_135, "diagonal target got {deg}");
    }

    #[test]
    fn best_rotation_always_in_range() {
        let pts = vec![0.0, 0.0, 3.0, 1.0, -1.0, 2.0, 0.5, -2.0];
        for &target in &[0.1_f32, 0.5, 1.0, 2.0, 10.0] {
            let deg = best_rotation_deg_for_aspect(&pts, target);
            assert!((0.0..180.0).contains(&deg), "deg {deg} out of [0,180)");
        }
    }
}
