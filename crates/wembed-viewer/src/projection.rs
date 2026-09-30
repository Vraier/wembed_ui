//! d-dimensional → 2D display projection.
//!
//! The render stack (scene, camera, framing, picking, shaders) is strictly 2D. Embeddings
//! with dimension > 2 pass through a [`Projector`] right where the producer's positions
//! enter the app/headless tick: raw d-dim coords in, projected 2D view out. Everything
//! downstream keeps its 2D contract untouched; the raw coords are kept around only for
//! re-projection, depth shading, and full-fidelity `.emb` saves.
//!
//! Two modes ([`ProjectionMode`]): `first-two` displays coords 0/1 verbatim; `pca`
//! displays the mean-centered projection onto the top-2 principal components, fitted on a
//! deterministic stride sample of at most [`MAX_FIT_SAMPLE`] vertices (fit is O(s·d²),
//! applying it is O(n·d) — both strictly cheaper than the embedder step that produced the
//! positions, so live runs never bottleneck here).
//!
//! Stability: the basis refits on every positions update, and each new basis is aligned
//! to the previous one (axis swap + sign flip) so the live view doesn't mirror or spin
//! when eigenvectors drift. An eigenvalue crossing still reorders the axes eventually —
//! that shows as one discrete reorientation, not per-frame jitter.
//!
//! Depth shading: per-vertex residual `r = ‖(x − mean) − P·Pᵀ(x − mean)‖` — the distance
//! the 2D view fails to show. Robustness comes from quantile normalization: the sample's
//! q05..q95 residual range maps to [0, 1] (clamped), recomputed only on refit so the
//! shading doesn't flicker between steps.

use serde::{Deserialize, Serialize};

/// How embeddings with dimension > 2 map into the 2D view (the `[visual] projection` TOML
/// key + the UI dropdown; no CLI flag). Irrelevant at dimension 2 (identity either way).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectionMode {
    /// Display coords 0/1 verbatim, dropping the rest. Cheap and stable, but nothing
    /// concentrates variance there — usually a poor view. Kept as the fast/debug option.
    FirstTwo,
    /// Mean-centered projection onto the top-2 principal components of a vertex sample.
    #[default]
    Pca,
}

/// Cap on the number of vertices the PCA/quantile fit looks at. Deterministic stride
/// sampling (not RNG) so identical inputs render identically — the snakemake pipeline
/// depends on reproducible output.
pub const MAX_FIT_SAMPLE: usize = 10_000;

const POWER_ITERS: usize = 300;
const POWER_TOL: f64 = 1e-12;

struct Fitted {
    /// Sample mean, f32 for the O(n·d) hot loops (fit runs in f64).
    mean: Vec<f32>,
    /// Orthonormal projection axes. `None` = canonical e0/e1 (first-two mode), where the
    /// display shows raw coords and only depth uses `mean`.
    axes: Option<[Vec<f32>; 2]>,
    /// Fraction of total variance the displayed plane captures, for the UI readout.
    explained: f32,
    /// Residual quantile bounds (q05, 1/(q95−q05)) for depth normalization.
    /// `depth_inv_range == 0` marks a degenerate spread → depth renders as 0.
    depth_lo: f32,
    depth_inv_range: f32,
}

/// Owns the projection state for one topology. Rebuild on topology change (vertex count
/// and positions layout change across layered-embedder layer transitions).
pub struct Projector {
    dim: usize,
    mode: ProjectionMode,
    fitted: Option<Fitted>,
}

impl Projector {
    pub fn new(dim: usize, mode: ProjectionMode) -> Self {
        Self {
            dim,
            mode,
            fitted: None,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// True when the projection is a passthrough and depth shading is meaningless.
    pub fn is_identity(&self) -> bool {
        self.dim <= 2
    }

    /// Switch modes; returns whether anything changed (caller re-projects + refits camera).
    pub fn set_mode(&mut self, mode: ProjectionMode) -> bool {
        if self.mode == mode {
            return false;
        }
        self.mode = mode;
        self.fitted = None;
        true
    }

    /// Fraction of total variance the displayed plane captures (both modes), `None` until
    /// fitted or when the projection is an identity.
    pub fn explained_variance(&self) -> Option<f32> {
        if self.is_identity() {
            return None;
        }
        self.fitted.as_ref().map(|f| f.explained)
    }

    /// Project raw d-dim `positions` (flat row-major) into `out` as flat 2D, refitting
    /// the basis first (every call — the sampled fit is cheaper than the embedder step
    /// that produced the positions; sign/swap alignment keeps consecutive fits stable).
    /// Depth values and quantile bounds move with every refit, so the caller should
    /// recompute the depth channel after each call.
    pub fn update_view(&mut self, positions: &[f32], out: &mut Vec<f32>) {
        if self.is_identity() {
            out.clear();
            out.extend_from_slice(positions);
            return;
        }
        debug_assert_eq!(positions.len() % self.dim, 0);
        self.fit(positions);
        self.project(positions, out);
    }

    /// Per-vertex depth-shade values in [0, 1] (0 = in-plane, 1 = at/beyond the q95
    /// residual). All zeros for identity projections or before the first fit.
    pub fn depth_values(&self, positions: &[f32], out: &mut Vec<f32>) {
        let dim = self.dim;
        let n = positions.len() / dim;
        out.clear();
        if self.is_identity() {
            out.resize(n, 0.0);
            return;
        }
        let Some(f) = &self.fitted else {
            out.resize(n, 0.0);
            return;
        };
        if f.depth_inv_range == 0.0 {
            out.resize(n, 0.0);
            return;
        }
        out.reserve(n);
        for v in positions.chunks_exact(dim) {
            let r = residual(v, &f.mean, f.axes.as_ref());
            out.push(((r - f.depth_lo) * f.depth_inv_range).clamp(0.0, 1.0));
        }
    }

    fn project(&self, positions: &[f32], out: &mut Vec<f32>) {
        let dim = self.dim;
        let n = positions.len() / dim;
        out.clear();
        out.reserve(n * 2);
        match self.fitted.as_ref().and_then(|f| f.axes.as_ref()) {
            // first-two: raw coords, comparable 1:1 with the saved .emb
            None => {
                for v in positions.chunks_exact(dim) {
                    out.push(v[0]);
                    out.push(v[1]);
                }
            }
            Some([a0, a1]) => {
                let mean = &self.fitted.as_ref().unwrap().mean;
                for v in positions.chunks_exact(dim) {
                    let mut y0 = 0.0_f32;
                    let mut y1 = 0.0_f32;
                    for k in 0..dim {
                        let c = v[k] - mean[k];
                        y0 += c * a0[k];
                        y1 += c * a1[k];
                    }
                    out.push(y0);
                    out.push(y1);
                }
            }
        }
    }

    fn fit(&mut self, positions: &[f32]) {
        let dim = self.dim;
        let n = positions.len() / dim;
        if n == 0 {
            self.fitted = None;
            return;
        }
        let stride = n.div_ceil(MAX_FIT_SAMPLE);
        let sample: Vec<&[f32]> = (0..n)
            .step_by(stride)
            .map(|i| &positions[i * dim..(i + 1) * dim])
            .collect();
        let s = sample.len() as f64;

        let mut mean = vec![0.0_f64; dim];
        for v in &sample {
            for k in 0..dim {
                mean[k] += v[k] as f64;
            }
        }
        for m in &mut mean {
            *m /= s;
        }

        let axes = match self.mode {
            ProjectionMode::FirstTwo => None,
            ProjectionMode::Pca => pca_axes(&sample, &mean, dim),
        };

        // total variance (covariance trace) for the explained-variance readout
        let mut trace = 0.0_f64;
        for v in &sample {
            for k in 0..dim {
                let c = v[k] as f64 - mean[k];
                trace += c * c;
            }
        }
        trace /= s;

        let mean_f32: Vec<f32> = mean.iter().map(|m| *m as f32).collect();
        let mut axes_f32: Option<[Vec<f32>; 2]> = axes.map(|[a0, a1]| {
            [
                a0.iter().map(|x| *x as f32).collect(),
                a1.iter().map(|x| *x as f32).collect(),
            ]
        });

        // continuity across refits: match the new pair to the old (swap on eigenvalue
        // crossing), then sign-flip so dot(new, old) >= 0 — otherwise the live view
        // mirrors/spins between publishes
        if let (Some(new), Some(prev)) = (
            axes_f32.as_mut(),
            self.fitted.as_ref().and_then(|f| f.axes.as_ref()),
        ) {
            align_axes(new, prev);
        }

        // displayed variance: PCA plane = Rayleigh quotients of the axes; first-two =
        // Var(x0) + Var(x1). Both are exactly the variance captured by the display plane.
        let plane_var: f64 = match &axes_f32 {
            Some([a0, a1]) => {
                variance_along(&sample, &mean, a0) + variance_along(&sample, &mean, a1)
            }
            None => {
                let mut v = 0.0_f64;
                for p in &sample {
                    for k in 0..2 {
                        let c = p[k] as f64 - mean[k];
                        v += c * c;
                    }
                }
                v / s
            }
        };
        let explained = if trace > 0.0 {
            (plane_var / trace).clamp(0.0, 1.0) as f32
        } else {
            1.0
        };

        // residual quantile bounds from the same sample
        let mut residuals: Vec<f32> = sample
            .iter()
            .map(|v| residual(v, &mean_f32, axes_f32.as_ref()))
            .collect();
        residuals.sort_by(f32::total_cmp);
        let q = |frac: f64| -> f32 {
            let idx = ((residuals.len() - 1) as f64 * frac).round() as usize;
            residuals[idx]
        };
        // spread below 0.1% of q95 = effectively constant residuals → shading would be
        // amplified noise, disable it
        let (lo, hi) = (q(0.05), q(0.95));
        let depth_inv_range = if hi - lo > 1e-3 * hi.abs().max(f32::MIN_POSITIVE) {
            1.0 / (hi - lo)
        } else {
            0.0
        };

        self.fitted = Some(Fitted {
            mean: mean_f32,
            axes: axes_f32,
            explained,
            depth_lo: lo,
            depth_inv_range,
        });
    }
}

/// Residual distance from the display plane: `‖(x−mean) − P·Pᵀ·(x−mean)‖`, computed as
/// the norm of the per-component reconstruction error — NOT via `‖c‖² − y0² − y1²`,
/// which catastrophically cancels in f32 when in-plane coords dwarf the residual.
/// `axes == None` means canonical e0/e1: the residual is just the hidden coords 2..d.
fn residual(v: &[f32], mean: &[f32], axes: Option<&[Vec<f32>; 2]>) -> f32 {
    let dim = v.len();
    let mut acc = 0.0_f32;
    match axes {
        None => {
            for k in 2..dim {
                let c = v[k] - mean[k];
                acc += c * c;
            }
        }
        Some([a0, a1]) => {
            let mut y0 = 0.0_f32;
            let mut y1 = 0.0_f32;
            for k in 0..dim {
                let c = v[k] - mean[k];
                y0 += c * a0[k];
                y1 += c * a1[k];
            }
            for k in 0..dim {
                let c = v[k] - mean[k] - y0 * a0[k] - y1 * a1[k];
                acc += c * c;
            }
        }
    }
    acc.sqrt()
}

/// Sample variance along a unit direction (Rayleigh quotient without materializing C).
fn variance_along(sample: &[&[f32]], mean: &[f64], axis: &[f32]) -> f64 {
    let mut acc = 0.0_f64;
    for v in sample {
        let mut y = 0.0_f64;
        for k in 0..axis.len() {
            y += (v[k] as f64 - mean[k]) * axis[k] as f64;
        }
        acc += y * y;
    }
    acc / sample.len() as f64
}

/// Top-2 eigenvectors of the sample covariance via power iteration + deflation.
/// Returns `None` on a degenerate cloud (all points identical) — caller falls back to
/// canonical axes.
fn pca_axes(sample: &[&[f32]], mean: &[f64], dim: usize) -> Option<[Vec<f64>; 2]> {
    let s = sample.len() as f64;
    // covariance, upper triangle accumulated then mirrored — O(s·d²/2), the fit's hot part
    let mut cov = vec![0.0_f64; dim * dim];
    let mut centered = vec![0.0_f64; dim];
    for v in sample {
        for k in 0..dim {
            centered[k] = v[k] as f64 - mean[k];
        }
        for i in 0..dim {
            let ci = centered[i];
            for j in i..dim {
                cov[i * dim + j] += ci * centered[j];
            }
        }
    }
    for i in 0..dim {
        for j in i..dim {
            let x = cov[i * dim + j] / s;
            cov[i * dim + j] = x;
            cov[j * dim + i] = x;
        }
    }

    let (v1, l1) = power_iterate(&cov, dim, None)?;
    let (mut v2, _) = power_iterate(&cov, dim, Some((&v1, l1)))
        .unwrap_or_else(|| (orthogonal_fallback(&v1), 0.0));
    // explicit re-orthogonalization — deflation alone leaves O(tol) leakage
    let d = dot(&v1, &v2);
    for k in 0..dim {
        v2[k] -= d * v1[k];
    }
    if !normalize(&mut v2) {
        v2 = orthogonal_fallback(&v1);
    }
    Some([v1, v2])
}

/// Power iteration on `cov`, optionally deflating a known eigenpair. Deterministic start
/// vector (no RNG — reproducible renders). Returns `None` when the iterate collapses,
/// i.e. the (deflated) matrix is effectively zero.
fn power_iterate(
    cov: &[f64],
    dim: usize,
    deflate: Option<(&[f64], f64)>,
) -> Option<(Vec<f64>, f64)> {
    // mild index ramp breaks ties on symmetric clouds; any fixed start orthogonal to the
    // dominant eigenvector would stall, and this one never is in practice
    let mut v: Vec<f64> = (0..dim).map(|k| 1.0 + k as f64 * 0.001).collect();
    normalize(&mut v);
    let mut w = vec![0.0_f64; dim];
    for _ in 0..POWER_ITERS {
        for (i, wi) in w.iter_mut().enumerate() {
            let row = &cov[i * dim..(i + 1) * dim];
            *wi = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        }
        if let Some((u, lu)) = deflate {
            let d = dot(u, &v);
            for k in 0..dim {
                w[k] -= lu * d * u[k];
            }
        }
        if !normalize(&mut w) {
            return None;
        }
        let advanced = 1.0 - dot(&w, &v).abs();
        std::mem::swap(&mut v, &mut w);
        if advanced < POWER_TOL {
            break;
        }
    }
    // Rayleigh quotient for the eigenvalue
    let mut lambda = 0.0_f64;
    for i in 0..dim {
        let row = &cov[i * dim..(i + 1) * dim];
        let cv: f64 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        lambda += v[i] * cv;
    }
    if let Some((u, lu)) = deflate {
        let d = dot(u, &v);
        lambda -= lu * d * d;
    }
    if lambda <= 1e-12 {
        return None;
    }
    Some((v, lambda))
}

/// A unit vector orthogonal to `v`: take the canonical basis vector `v` points at least,
/// Gram-Schmidt it. Used when the cloud has < 2 directions of real variance.
fn orthogonal_fallback(v: &[f64]) -> Vec<f64> {
    let dim = v.len();
    let k_min = (0..dim)
        .min_by(|a, b| v[*a].abs().total_cmp(&v[*b].abs()))
        .unwrap();
    let mut out = vec![0.0_f64; dim];
    out[k_min] = 1.0;
    let d = dot(v, &out);
    for k in 0..dim {
        out[k] -= d * v[k];
    }
    normalize(&mut out);
    out
}

fn align_axes(new: &mut [Vec<f32>; 2], prev: &[Vec<f32>; 2]) {
    if new[0].len() != prev[0].len() {
        return;
    }
    let d = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let straight = d(&new[0], &prev[0]).abs() + d(&new[1], &prev[1]).abs();
    let crossed = d(&new[0], &prev[1]).abs() + d(&new[1], &prev[0]).abs();
    if crossed > straight {
        new.swap(0, 1);
    }
    for (axis, old) in new.iter_mut().zip(prev.iter()) {
        if d(axis, old) < 0.0 {
            for x in axis.iter_mut() {
                *x = -*x;
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Normalize in place; false when the norm is (numerically) zero.
fn normalize(v: &mut [f64]) -> bool {
    let norm = dot(v, v).sqrt();
    if norm <= 1e-30 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-noise without a rand dependency.
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // roughly [-1, 1)
    }

    /// 4D cloud with variance overwhelmingly along two known orthogonal directions.
    fn anisotropic_cloud(n: usize) -> Vec<f32> {
        // u = (1,1,0,0)/√2 with spread 100, w = (0,0,1,-1)/√2 with spread 10, noise 0.01
        let inv = 1.0_f32 / 2.0_f32.sqrt();
        let (u, w) = ([inv, inv, 0.0, 0.0], [0.0, 0.0, inv, -inv]);
        let mut state = 42_u64;
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            let (t, s) = (100.0 * lcg(&mut state), 10.0 * lcg(&mut state));
            for k in 0..4 {
                out.push(t * u[k] + s * w[k] + 0.01 * lcg(&mut state) + 5.0);
            }
        }
        out
    }

    #[test]
    fn pca_recovers_dominant_plane() {
        let positions = anisotropic_cloud(2000);
        let mut p = Projector::new(4, ProjectionMode::Pca);
        let mut view = Vec::new();
        p.update_view(&positions, &mut view);

        // the top-2 plane captures essentially all variance
        let explained = p.explained_variance().expect("fitted");
        assert!(explained > 0.999, "explained variance {explained}");

        // axis 0 must be (anti)parallel to the dominant direction u
        let inv = 1.0_f32 / 2.0_f32.sqrt();
        let axes = p.fitted.as_ref().unwrap().axes.as_ref().unwrap();
        let d0: f32 = axes[0]
            .iter()
            .zip([inv, inv, 0.0, 0.0])
            .map(|(a, b)| a * b)
            .sum();
        assert!(d0.abs() > 0.999, "axis0·u = {d0}");
    }

    #[test]
    fn refit_alignment_prevents_flips() {
        let positions = anisotropic_cloud(2000);
        let mut p = Projector::new(4, ProjectionMode::Pca);
        let mut view = Vec::new();
        p.update_view(&positions, &mut view);
        let before: Vec<Vec<f32>> = {
            let axes = p.fitted.as_ref().unwrap().axes.as_ref().unwrap();
            vec![axes[0].clone(), axes[1].clone()]
        };

        // perturb and force a refit — axes must stay aligned (dot > 0), never mirror
        let perturbed: Vec<f32> = {
            let mut state = 7_u64;
            positions
                .iter()
                .map(|x| x + 0.05 * lcg(&mut state))
                .collect()
        };
        p.update_view(&perturbed, &mut view);
        let axes = p.fitted.as_ref().unwrap().axes.as_ref().unwrap();
        for (new, old) in axes.iter().zip(before.iter()) {
            let d: f32 = new.iter().zip(old).map(|(a, b)| a * b).sum();
            assert!(d > 0.9, "axis flipped or wandered: dot = {d}");
        }
    }

    #[test]
    fn first_two_is_verbatim_and_depth_tracks_hidden_coords() {
        // continuous hidden-coord spread, plus one far-out-of-plane vertex; in-plane
        // coords are large so this doubles as a cancellation regression check
        let mut state = 3_u64;
        let mut positions = Vec::new();
        for i in 0..200 {
            let h = 0.5 * lcg(&mut state);
            positions.extend_from_slice(&[i as f32, -(i as f32), h, -h]);
        }
        positions.extend_from_slice(&[0.0, 0.0, 50.0, 0.0]);

        let mut p = Projector::new(4, ProjectionMode::FirstTwo);
        let mut view = Vec::new();
        p.update_view(&positions, &mut view);
        // display is raw coords 0/1
        assert_eq!(view[..4], [0.0, -0.0, 1.0, -1.0]);

        let mut depth = Vec::new();
        p.depth_values(&positions, &mut depth);
        assert_eq!(depth[200], 1.0, "far outlier clamps to 1");
        let mean_typical: f32 = depth[..200].iter().sum::<f32>() / 200.0;
        assert!(mean_typical < 0.7, "typical in-plane depth {mean_typical}");
    }

    #[test]
    fn dim_two_is_pure_passthrough() {
        let positions = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut p = Projector::new(2, ProjectionMode::Pca);
        let mut view = Vec::new();
        p.update_view(&positions, &mut view);
        assert_eq!(view, positions);
        let mut depth = Vec::new();
        p.depth_values(&positions, &mut depth);
        assert_eq!(depth, vec![0.0, 0.0]);
        assert_eq!(p.explained_variance(), None);
    }
}
