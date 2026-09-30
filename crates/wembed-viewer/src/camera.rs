//! Fit-to-bounds 2D camera.
//!
//! Interface is intentionally small: `fit_to_quantile_flat` to derive the framing from a point
//! cloud, `view_proj` to build the aspect-corrected orthographic matrix for a given window
//! size, pan/zoom for interaction, and `fit_half`/`zoom` so sizes (point radius, edge width)
//! can be chosen at the fitted framing and then follow the zoom like a crop.

use glam::{Mat2, Mat4, Vec2};

pub struct Camera {
    center: [f32; 2],
    half: f32,
    /// `half` of the fitted framing, which dot and edge sizes are chosen for — so zooming
    /// away from it magnifies the picture instead of re-laying it out. Pan/zoom keep it.
    fit_half: f32,
    /// Rotation applied to the world before projection, in radians (positive = CCW). All
    /// interactive input (pan, zoom, pick) accounts for it via inverse rotation, so
    /// switching rotation feels natural regardless of angle.
    rotation: f32,
    /// Mirror the rendered image along x / y. Applied in *clip space* (after rotation),
    /// so "flip x" always means "mirror the picture left-right" regardless of the current
    /// rotation. Inverse-applied in `pixel_to_world` / `pan_pixels`, same contract as
    /// `rotation`.
    flip: [bool; 2],
}

/// Extend framing bounds toward the true extremes, capped at `buffer` fraction of the
/// quantile span beyond each quantile bound. If the extremes are within that distance the
/// whole graph is framed; otherwise outliers get clipped at the buffer edge.
fn apply_buffer(true_min: f32, true_max: f32, q_lo: f32, q_hi: f32, buffer: f32) -> (f32, f32) {
    let span = (q_hi - q_lo).max(0.0);
    let slack = span * buffer;
    let lo = true_min.max(q_lo - slack);
    let hi = true_max.min(q_hi + slack);
    (lo, hi)
}

/// Two `O(N)` partial-sorts to extract the low + high quantile bounds in place. Unwraps
/// `partial_cmp` — positions from a well-behaved embedder shouldn't contain NaN, and if they
/// do we have a bug to surface, not silence.
fn quantile_bounds(v: &mut [f32], q_lo: f32, q_hi: f32) -> (f32, f32) {
    let n = v.len();
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).expect("NaN in positions");
    let lo_idx = ((q_lo * n as f32) as usize).min(n - 1);
    let hi_idx = ((q_hi * n as f32) as usize).min(n - 1);
    v.select_nth_unstable_by(lo_idx, cmp);
    let lo = v[lo_idx];
    v.select_nth_unstable_by(hi_idx, cmp);
    let hi = v[hi_idx];
    (lo, hi)
}

impl Camera {
    /// Frame the point cloud using per-axis quantile bounds, expanded to include the true
    /// extremes when they're within `buffer` fraction of the quantile span. So: no outliers
    /// → the whole graph fits; outliers → they get clipped at `buffer * span` beyond the
    /// quantile. A 20% render margin is applied on top.
    ///
    /// Uses `select_nth_unstable` on the full position arrays — O(N) per axis, but with
    /// non-trivial constants at 1M+ vertices. Two `Vec<f32>` allocations per call scale with
    /// N; if that ever bites we swap for a subsampled quantile.
    ///
    /// Flat interleaved input `[x0, y0, x1, y1, ...]`, matching what wembed's FFI produces.
    pub fn fit_to_quantile_flat(positions: &[f32], q_lo: f32, q_hi: f32, buffer: f32) -> Self {
        debug_assert!(positions.len().is_multiple_of(2));
        debug_assert!(0.0 <= q_lo && q_lo < q_hi && q_hi <= 1.0);
        debug_assert!(buffer >= 0.0);

        let n = positions.len() / 2;
        if n == 0 {
            return Self {
                center: [0.0, 0.0],
                half: 1.0,
                fit_half: 1.0,
                rotation: 0.0,
                flip: [false; 2],
            };
        }

        // Extract per-axis arrays and true extremes in a single pass. `select_nth_unstable`
        // later disturbs `xs`/`ys` ordering, so grab true_min/max here while we still can.
        let mut xs: Vec<f32> = Vec::with_capacity(n);
        let mut ys: Vec<f32> = Vec::with_capacity(n);
        let mut true_min = [f32::INFINITY; 2];
        let mut true_max = [f32::NEG_INFINITY; 2];
        for xy in positions.chunks_exact(2) {
            xs.push(xy[0]);
            ys.push(xy[1]);
            true_min[0] = true_min[0].min(xy[0]);
            true_min[1] = true_min[1].min(xy[1]);
            true_max[0] = true_max[0].max(xy[0]);
            true_max[1] = true_max[1].max(xy[1]);
        }

        let (q_lo_x, q_hi_x) = quantile_bounds(&mut xs, q_lo, q_hi);
        let (q_lo_y, q_hi_y) = quantile_bounds(&mut ys, q_lo, q_hi);

        let (min_x, max_x) = apply_buffer(true_min[0], true_max[0], q_lo_x, q_hi_x, buffer);
        let (min_y, max_y) = apply_buffer(true_min[1], true_max[1], q_lo_y, q_hi_y, buffer);

        let center = [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5];
        let half = ((max_x - min_x).max(max_y - min_y) * 0.5).max(1e-3) * 1.2;
        Self {
            center,
            half,
            fit_half: half,
            rotation: 0.0,
            flip: [false; 2],
        }
    }

    /// This fit, re-aimed at previously-captured framing — the `[camera]` config section.
    /// Counterpart of `center()` / `half()`, so a TOML-exported framing restores exactly; the
    /// fit stays the size reference, so the restored view is sized like the session it was
    /// saved from. `half` is floored so a hand-edited zero can't collapse the projection.
    pub fn with_view(mut self, center: [f32; 2], half: f32) -> Self {
        self.center = center;
        self.half = half.max(1e-6);
        self
    }

    /// Set the camera rotation (radians, positive = CCW). Applied around the current
    /// center in `view_proj` and inverse-applied in `pixel_to_world` / `pan_pixels` so
    /// pointer input stays consistent regardless of angle.
    pub fn set_rotation(&mut self, radians: f32) {
        self.rotation = radians;
    }

    /// Mirror the output along x (left↔right) and/or y (top↔bottom). Clip-space, so the
    /// mirror is of the final image — independent of rotation. Inverse-applied to all
    /// pointer input like rotation is.
    pub fn set_flip(&mut self, flip_x: bool, flip_y: bool) {
        self.flip = [flip_x, flip_y];
    }

    /// Current rotation in radians. The TOML export reads `VisualState::rotation_deg`
    /// instead (exact degrees, no radian round-trip); kept for future callers.
    #[allow(dead_code)]
    pub fn rotation(&self) -> f32 {
        self.rotation
    }

    /// World-space center of the framing. Captured by the TOML export.
    pub fn center(&self) -> [f32; 2] {
        self.center
    }

    /// Half-extent of the framed span along the larger axis (world units) — the zoom
    /// level, as captured by the TOML export.
    pub fn half(&self) -> f32 {
        self.half
    }

    /// Reset rotation to zero. Not currently invoked — App preserves rotation across
    /// topology swaps via `VisualState.rotation_deg`. Kept for a future "reset view"
    /// button.
    #[allow(dead_code)]
    pub fn reset_rotation(&mut self) {
        self.rotation = 0.0;
    }

    /// `half` of the fitted framing (see the field).
    pub fn fit_half(&self) -> f32 {
        self.fit_half
    }

    /// Magnification relative to the fitted framing (> 1 = zoomed in).
    pub fn zoom(&self) -> f32 {
        self.fit_half / self.half
    }

    /// World units per screen pixel. Uniform across x/y because `view_proj` corrects for
    /// aspect. Used by point-size heuristics to translate between pixel and world radii.
    pub fn world_per_pixel(&self, width: u32, height: u32) -> f32 {
        let aspect = width as f32 / height.max(1) as f32;
        // Height of the viewport in world units (aspect-corrected).
        let hy = if aspect >= 1.0 {
            self.half
        } else {
            self.half / aspect
        };
        (2.0 * hy) / height.max(1) as f32
    }

    /// Aspect-corrected orthographic projection. Depth range is [0, 1] to match wgpu's
    /// clip-space convention (Vulkan/DirectX style).
    ///
    /// When `rotation != 0`, the world is rotated around the camera center before
    /// projection: `clip = ortho * translate(center) * rotate(θ) * translate(-center) *
    /// world`. `pixel_to_world` applies the inverse so mouse input stays consistent.
    pub fn view_proj(&self, width: u32, height: u32) -> Mat4 {
        let (hx, hy) = self.half_extents(width, height);
        let (l, r, b, t) = (
            self.center[0] - hx,
            self.center[0] + hx,
            self.center[1] - hy,
            self.center[1] + hy,
        );
        let mut ortho = Mat4::from_cols_array_2d(&[
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (t - b), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-(r + l) / (r - l), -(t + b) / (t - b), 0.0, 1.0],
        ]);
        // Mirror in clip space (NDC is centered on 0, so a sign change mirrors the
        // image about the screen center). Outermost transform → the *rendered* picture
        // flips, whatever the rotation underneath does.
        if self.flip[0] || self.flip[1] {
            let s = glam::vec3(
                if self.flip[0] { -1.0 } else { 1.0 },
                if self.flip[1] { -1.0 } else { 1.0 },
                1.0,
            );
            ortho = Mat4::from_scale(s) * ortho;
        }
        if self.rotation == 0.0 {
            return ortho;
        }
        let cx = self.center[0];
        let cy = self.center[1];
        let to_origin = Mat4::from_translation(glam::vec3(-cx, -cy, 0.0));
        let rotate = Mat4::from_rotation_z(self.rotation);
        let from_origin = Mat4::from_translation(glam::vec3(cx, cy, 0.0));
        ortho * from_origin * rotate * to_origin
    }

    /// Aspect-corrected half-extents in world units. `view_proj` and `pixel_to_world`
    /// share this so the two stay consistent. Also consulted by the export-frame overlay
    /// (`app::export_frame_pixel_rect`) to compute where the export target's viewport lands
    /// on the current preview canvas.
    pub fn half_extents(&self, width: u32, height: u32) -> (f32, f32) {
        let aspect = width as f32 / height.max(1) as f32;
        if aspect >= 1.0 {
            (self.half * aspect, self.half)
        } else {
            (self.half, self.half / aspect)
        }
    }

    /// Screen-pixel coordinate → world coordinate. Screen y grows downward; world y grows up.
    /// Accounts for the current rotation: after the standard NDC-scaled offset from center,
    /// we apply the inverse of view_proj's rotation so that "pixel under cursor" always
    /// maps to the world point actually rendered there.
    pub fn pixel_to_world(&self, mx: f32, my: f32, width: u32, height: u32) -> [f32; 2] {
        let (hx, hy) = self.half_extents(width, height);
        let mut ndc_x = 2.0 * mx / width.max(1) as f32 - 1.0;
        let mut ndc_y = 1.0 - 2.0 * my / height.max(1) as f32;
        // Undo the clip-space mirror first — it's the outermost transform in view_proj,
        // so it inverts first on the way back.
        if self.flip[0] {
            ndc_x = -ndc_x;
        }
        if self.flip[1] {
            ndc_y = -ndc_y;
        }
        let offset = Vec2::new(ndc_x * hx, ndc_y * hy);
        // Inverse of the +θ rotation applied in view_proj.
        let rotated = Mat2::from_angle(-self.rotation) * offset;
        [self.center[0] + rotated.x, self.center[1] + rotated.y]
    }

    /// Shift the center by a screen-pixel drag delta. Screen y grows downward; a drag of
    /// `dx, dy` moves the scene by that many pixels under the cursor (center moves the
    /// opposite direction). Rotation-aware: the screen-space delta is inverse-rotated
    /// before being applied to `center`, so panning always feels like "drag the world
    /// in the direction the mouse moves" regardless of camera angle.
    pub fn pan_pixels(&mut self, dx: f32, dy: f32, width: u32, height: u32) {
        let wpp = self.world_per_pixel(width, height);
        let mut screen_delta = Vec2::new(dx * wpp, -dy * wpp);
        // Same inverse chain as pixel_to_world: undo the clip-space mirror, then the
        // rotation.
        if self.flip[0] {
            screen_delta.x = -screen_delta.x;
        }
        if self.flip[1] {
            screen_delta.y = -screen_delta.y;
        }
        let world_delta = Mat2::from_angle(-self.rotation) * screen_delta;
        self.center[0] -= world_delta.x;
        self.center[1] -= world_delta.y;
    }

    /// Scale `half` by `factor` while keeping the world point under the cursor fixed. Zoom
    /// in: `factor < 1`. Zoom out: `factor > 1`. `half` is floored at a small epsilon so
    /// runaway zoom-in can't collapse the view.
    pub fn zoom_at_pixel(&mut self, factor: f32, mx: f32, my: f32, width: u32, height: u32) {
        let before = self.pixel_to_world(mx, my, width, height);
        self.half = (self.half * factor).max(1e-6);
        let after = self.pixel_to_world(mx, my, width, height);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    /// Interleave an `(x, y)` list into the flat buffer the camera expects.
    fn flat(pts: &[(f32, f32)]) -> Vec<f32> {
        pts.iter().flat_map(|&(x, y)| [x, y]).collect()
    }

    // ---- fit_to_quantile_flat ----

    #[test]
    fn fit_empty_positions_is_default_camera() {
        let cam = Camera::fit_to_quantile_flat(&[], 0.005, 0.995, 0.1);
        // Center at origin, no panic.
        let c = cam.center;
        assert!(close(c[0], 0.0) && close(c[1], 0.0));
        assert!(cam.half > 0.0);
    }

    #[test]
    fn fit_single_point_has_nonzero_half() {
        // Single point → zero span → `half` floored at the 1e-3 floor (times the 1.2 margin).
        let cam = Camera::fit_to_quantile_flat(&flat(&[(3.0, 4.0)]), 0.005, 0.995, 0.1);
        assert!(cam.half >= 1e-3, "half {} below floor", cam.half);
        assert!(close(cam.center[0], 3.0) && close(cam.center[1], 4.0));
    }

    #[test]
    fn fit_no_outliers_includes_true_extremes() {
        // A tight symmetric cloud with no outliers: the framing must reach the true min/max.
        let pts = flat(&[
            (-10.0, -10.0),
            (10.0, 10.0),
            (-5.0, 5.0),
            (5.0, -5.0),
            (0.0, 0.0),
        ]);
        let cam = Camera::fit_to_quantile_flat(&pts, 0.005, 0.995, 0.1);
        // half = max_span * 0.5 * 1.2. Span here is 20 on both axes → half = 12.
        assert!(close(cam.center[0], 0.0) && close(cam.center[1], 0.0));
        assert!(close(cam.half, 12.0), "half {}", cam.half);
    }

    #[test]
    fn fit_one_extreme_outlier_is_clipped_at_buffer() {
        // 400 points densely in [0,1] on x, plus one far outlier at x=1000. With this many
        // inliers the 99.5th percentile lands among them, so the outlier is beyond the
        // quantile bound and gets clipped at `buffer * span` well short of 1000.
        let mut pts: Vec<(f32, f32)> = (0..400).map(|i| (i as f32 / 399.0, 0.0)).collect();
        pts.push((1000.0, 0.0));
        let cam = Camera::fit_to_quantile_flat(&flat(&pts), 0.005, 0.995, 0.1);
        // Without clipping, half would be ~600. With clipping it stays near the quantile
        // span (order-1), so assert the outlier did NOT dominate the framing.
        assert!(cam.half < 10.0, "outlier not clipped, half {}", cam.half);
    }

    // ---- quantile_bounds / apply_buffer (private free fns, same-file access) ----

    #[test]
    fn quantile_bounds_exact_on_known_array() {
        let mut v: Vec<f32> = (0..10).map(|i| i as f32).collect();
        // n=10, q_lo=0.2 → lo_idx=2 → value 2; q_hi=0.8 → hi_idx=8 → value 8.
        let (lo, hi) = quantile_bounds(&mut v, 0.2, 0.8);
        assert!(close(lo, 2.0), "lo {lo}");
        assert!(close(hi, 8.0), "hi {hi}");
    }

    #[test]
    fn apply_buffer_larger_than_outlier_distance_includes_extremes() {
        // quantile span [0, 10] → slack = 10 * 0.5 = 5. true extremes -3 and 12 fall
        // within [0-5, 10+5] = [-5, 15], so they're included verbatim.
        let (lo, hi) = apply_buffer(-3.0, 12.0, 0.0, 10.0, 0.5);
        assert!(close(lo, -3.0), "lo {lo}");
        assert!(close(hi, 12.0), "hi {hi}");
    }

    #[test]
    fn apply_buffer_clips_when_extreme_beyond_slack() {
        // span 10, slack = 1. true_min -100 is far beyond q_lo - slack = -1, so it clips
        // to -1; true_max 100 clips to 11.
        let (lo, hi) = apply_buffer(-100.0, 100.0, 0.0, 10.0, 0.1);
        assert!(close(lo, -1.0), "lo {lo}");
        assert!(close(hi, 11.0), "hi {hi}");
    }

    // ---- pixel/world round-trips ----

    /// A camera centered at (5, 7) with a predictable half. Symmetric cloud → center is
    /// exactly the midpoint regardless of quantile index rounding.
    fn centered_camera() -> Camera {
        let pts = flat(&[
            (5.0 - 4.0, 7.0 - 4.0),
            (5.0 + 4.0, 7.0 + 4.0),
            (5.0 - 4.0, 7.0 + 4.0),
            (5.0 + 4.0, 7.0 - 4.0),
            (5.0, 7.0),
        ]);
        Camera::fit_to_quantile_flat(&pts, 0.005, 0.995, 0.1)
    }

    #[test]
    fn pixel_to_world_of_center_is_camera_center_rotated() {
        for &deg in &[0.0_f32, 30.0, 90.0] {
            let mut cam = centered_camera();
            cam.set_rotation(deg.to_radians());
            let (w, h) = (800u32, 600u32);
            // The center is the rotation pivot, so the screen center still maps to it.
            let p = cam.pixel_to_world(w as f32 / 2.0, h as f32 / 2.0, w, h);
            assert!(
                close(p[0], cam.center[0]) && close(p[1], cam.center[1]),
                "deg {deg}: {p:?}"
            );
        }
    }

    #[test]
    fn zoom_at_pixel_keeps_world_under_cursor_fixed() {
        for &deg in &[0.0_f32, 30.0, 90.0] {
            let mut cam = centered_camera();
            cam.set_rotation(deg.to_radians());
            let (w, h) = (800u32, 600u32);
            let (mx, my) = (123.0_f32, 456.0_f32);
            let before = cam.pixel_to_world(mx, my, w, h);
            cam.zoom_at_pixel(0.5, mx, my, w, h);
            let after = cam.pixel_to_world(mx, my, w, h);
            assert!(
                close(before[0], after[0]) && close(before[1], after[1]),
                "deg {deg}: {before:?} vs {after:?}"
            );
        }
    }

    #[test]
    fn pan_pixels_moves_center_opposite_drag_rotation_aware() {
        // No rotation: dragging right/down moves the center left/up (world follows cursor).
        let mut cam = centered_camera();
        let (w, h) = (800u32, 600u32);
        let before = cam.center;
        let wpp = cam.world_per_pixel(w, h);
        cam.pan_pixels(10.0, 20.0, w, h);
        // screen_delta = (dx*wpp, -dy*wpp); center -= that. So center.x -= 10*wpp,
        // center.y -= -20*wpp = += 20*wpp.
        assert!(
            close(cam.center[0], before[0] - 10.0 * wpp),
            "x {}",
            cam.center[0]
        );
        assert!(
            close(cam.center[1], before[1] + 20.0 * wpp),
            "y {}",
            cam.center[1]
        );

        // With a 90° rotation the same screen drag moves the center along rotated axes.
        let mut cam2 = centered_camera();
        cam2.set_rotation(std::f32::consts::FRAC_PI_2);
        let before2 = cam2.center;
        cam2.pan_pixels(10.0, 0.0, w, h);
        // A pure-x screen drag under +90° rotation shifts the center in y (not x).
        assert!(
            close(cam2.center[0], before2[0]),
            "rotated x moved: {}",
            cam2.center[0]
        );
        assert!(
            (cam2.center[1] - before2[1]).abs() > EPS,
            "rotated y unchanged"
        );
    }

    #[test]
    fn view_proj_inverse_maps_pixel_to_world_back_to_ndc() {
        // Rotation × flip combinations — every pointer-input path (pick, zoom-at-cursor)
        // relies on pixel_to_world being the exact inverse of view_proj, and flips are
        // where sign errors hide.
        for &deg in &[0.0_f32, 30.0, 90.0] {
            for &(fx, fy) in &[(false, false), (true, false), (false, true), (true, true)] {
                let mut cam = centered_camera();
                cam.set_rotation(deg.to_radians());
                cam.set_flip(fx, fy);
                let (w, h) = (800u32, 600u32);
                let vp = cam.view_proj(w, h);
                // Pick a pixel, map to world, project back, and compare to that pixel's NDC.
                let (px, py) = (200.0_f32, 150.0_f32);
                let world = cam.pixel_to_world(px, py, w, h);
                let clip = vp * glam::vec4(world[0], world[1], 0.0, 1.0);
                let ndc = glam::vec2(clip.x / clip.w, clip.y / clip.w);
                let expect_ndc_x = 2.0 * px / w as f32 - 1.0;
                let expect_ndc_y = 1.0 - 2.0 * py / h as f32;
                assert!(
                    close(ndc.x, expect_ndc_x),
                    "deg {deg} flip ({fx},{fy}): ndc.x {} vs {expect_ndc_x}",
                    ndc.x
                );
                assert!(
                    close(ndc.y, expect_ndc_y),
                    "deg {deg} flip ({fx},{fy}): ndc.y {} vs {expect_ndc_y}",
                    ndc.y
                );
            }
        }
    }

    #[test]
    fn flip_mirrors_ndc_about_screen_center() {
        // A world point right/above of center must land left/below of it once flipped —
        // catches a flip that wires through as a no-op (round-trip alone can't: identity
        // passes it).
        let cam0 = centered_camera();
        let (w, h) = (800u32, 600u32);
        let world = glam::vec4(cam0.center()[0] + 1.0, cam0.center()[1] + 1.0, 0.0, 1.0);
        let base = cam0.view_proj(w, h) * world;
        let mut cam = centered_camera();
        cam.set_flip(true, true);
        let flipped = cam.view_proj(w, h) * world;
        assert!(close(flipped.x, -base.x), "{} vs {}", flipped.x, -base.x);
        assert!(close(flipped.y, -base.y), "{} vs {}", flipped.y, -base.y);
    }

    #[test]
    fn world_per_pixel_consistent_with_half_extents() {
        let cam = centered_camera();
        // Wide window: aspect >= 1. world_per_pixel = 2*half / height; half_extents.y = half.
        let (w, h) = (1600u32, 900u32);
        let wpp = cam.world_per_pixel(w, h);
        let (_hx, hy) = cam.half_extents(w, h);
        assert!(close(wpp, 2.0 * hy / h as f32), "wide wpp {wpp}");

        // Tall window: aspect < 1. hy = half / aspect; wpp still 2*hy/height.
        let (w2, h2) = (600u32, 1200u32);
        let wpp2 = cam.world_per_pixel(w2, h2);
        let (_hx2, hy2) = cam.half_extents(w2, h2);
        assert!(close(wpp2, 2.0 * hy2 / h2 as f32), "tall wpp {wpp2}");
    }
}
