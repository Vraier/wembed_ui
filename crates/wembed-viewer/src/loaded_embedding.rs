//! Load-once producer for a precomputed embedding.
//!
//! Reads a `.emb` CSV (per-vertex positions + weights) plus a wembed `.edg` file (edges),
//! synthesizes an [`embedder::Topology`] with the same shape a live embedder would emit,
//! and publishes it exactly once via [`TopologyProducer`]. Subsequent polls return
//! `Update::Nothing`. `is_paused` and `is_finished` are both `true` — the load-once path
//! has no run loop to pause or converge.
//!
//! `.emb` format (what `--output-embedding` saves):
//! `id,x1,...,xd,w` — dim inferred from the column count (`cols - 2`), must be >= 2 and
//! consistent across lines. No header line. Line order need not match `id`, but for the
//! initial load we sort by `id` so downstream code that assumes vertex-index = line-order
//! still works.
//!
//! `.edg` format: whitespace-separated `src dst` per line, comments starting with `#`.
//! Vertex ids referenced by edges must be < N (from the `.emb`).

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::embedder::{preprocess_topology, Topology, TopologyProducer, Update};

/// Parse a `.edg` file: `src dst` per line. Comments (`#`) and blank lines skipped.
/// Vertex ids beyond `n_vertices` are a hard error — surfaces mismatched .edg/.emb pairs.
pub fn parse_edges_file(path: &Path, n_vertices: usize) -> Result<Vec<[u32; 2]>> {
    let file = File::open(path).with_context(|| format!("open edges file {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let lineno = idx + 1;
        let raw = line.with_context(|| format!("read edges line {lineno}"))?;
        let s = raw.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let mut parts = s.split_ascii_whitespace();
        let src: u32 = parts
            .next()
            .and_then(|t| t.parse().ok())
            .with_context(|| format!("edges line {lineno}: missing/invalid src"))?;
        let dst: u32 = parts
            .next()
            .and_then(|t| t.parse().ok())
            .with_context(|| format!("edges line {lineno}: missing/invalid dst"))?;
        if src as usize >= n_vertices || dst as usize >= n_vertices {
            bail!(
                "edges line {lineno}: vertex id out of range for N={n_vertices} \
                 (got src={src}, dst={dst})"
            );
        }
        out.push([src, dst]);
    }
    Ok(out)
}

/// Parse a `.emb` CSV: `id,x1,...,xd,w` per line, dim inferred from the column count.
/// Returns `(positions, dim, weights)` where `positions[dim*i..dim*(i+1)]` are the coords
/// of vertex id `i` (row-major flat, matching the live embedder's layout) and `weights[i]`
/// is that vertex's raw weight.
pub fn parse_embedding_file(path: &Path) -> Result<(Vec<f32>, usize, Vec<f32>)> {
    let file =
        File::open(path).with_context(|| format!("open embedding file {}", path.display()))?;
    // Coords are collected flat in file order; a sort permutation over ids maps them into
    // id order at the end — avoids one Vec allocation per row.
    let mut ids: Vec<u32> = Vec::new();
    let mut weights_raw: Vec<f32> = Vec::new();
    let mut coords: Vec<f32> = Vec::new();
    let mut expected_dim: Option<usize> = None;
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let lineno = idx + 1;
        let raw = line.with_context(|| format!("read embedding line {lineno}"))?;
        let s = raw.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = s.split(',').collect();
        // Enforce a consistent dim across the file — surfaces mixed-dim lines.
        let dim = parts.len().saturating_sub(2); // exclude id + weight
        if let Some(d) = expected_dim {
            if dim != d {
                bail!(
                    "embedding line {lineno}: dim changed ({d} → {dim}); \
                     file must have a consistent column count"
                );
            }
        } else {
            if dim < 2 {
                bail!(
                    "embedding line {lineno}: expected id,x1,...,xd,w with d >= 2 \
                     but got {} columns",
                    parts.len()
                );
            }
            expected_dim = Some(dim);
        }
        let id: u32 = parts[0]
            .trim()
            .parse()
            .with_context(|| format!("embedding line {lineno}: bad id `{}`", parts[0]))?;
        for (k, tok) in parts[1..1 + dim].iter().enumerate() {
            let c: f32 = tok
                .trim()
                .parse()
                .with_context(|| format!("embedding line {lineno}: bad coord {} `{tok}`", k + 1))?;
            coords.push(c);
        }
        let w_tok = parts[1 + dim];
        let w: f32 = w_tok
            .trim()
            .parse()
            .with_context(|| format!("embedding line {lineno}: bad weight `{w_tok}`"))?;
        ids.push(id);
        weights_raw.push(w);
    }
    // Sort by id, then verify densely-packed [0, N). Reject sparse/duplicate ids —
    // downstream assumes vertex-index = position-index in flat arrays.
    let n = ids.len();
    let dim = expected_dim.unwrap_or(2);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| ids[i]);
    for (expected, &row) in order.iter().enumerate() {
        if ids[row] as usize != expected {
            bail!(
                "embedding file: expected vertex ids 0..N contiguous, but saw {} at position {}",
                ids[row],
                expected
            );
        }
    }
    let mut positions = Vec::with_capacity(n * dim);
    let mut weights = Vec::with_capacity(n);
    for &row in &order {
        positions.extend_from_slice(&coords[row * dim..(row + 1) * dim]);
        weights.push(weights_raw[row]);
    }
    Ok((positions, dim, weights))
}

/// Write a `.emb` file matching [`parse_embedding_file`]'s expected format. Emits one
/// line per vertex in id order: `i,x1,...,xd,w` — the FULL embedding, not the projected
/// view. Uses `{:.6}` precision for coordinates — enough to round-trip a `f32` for
/// typical embedding scales, keeps files reasonably small.
pub fn save_embedding_file(
    path: &Path,
    positions: &[f32],
    dim: usize,
    weights: &[f32],
) -> Result<()> {
    debug_assert!(dim >= 2 && positions.len().is_multiple_of(dim));
    let n = positions.len() / dim;
    if weights.len() != n {
        bail!(
            "save embedding: weights.len()={} but expected {n}",
            weights.len()
        );
    }
    let mut f =
        File::create(path).with_context(|| format!("create embedding file {}", path.display()))?;
    let mut line = String::new();
    for i in 0..n {
        use std::fmt::Write as _;
        line.clear();
        write!(line, "{i}").unwrap();
        for c in &positions[i * dim..(i + 1) * dim] {
            write!(line, ",{c:.6}").unwrap();
        }
        let w = weights[i];
        writeln!(f, "{line},{w}").with_context(|| format!("write embedding line {i}"))?;
    }
    Ok(())
}

/// Publish-once TopologyProducer for a precomputed embedding + graph.
pub struct LoadedEmbedding {
    topology: Option<Topology>,
    positions: Vec<f32>,
    /// Flipped to true on the first `poll` — subsequent polls return `Nothing`.
    published: bool,
}

impl LoadedEmbedding {
    /// Build from parsed positions/weights (from `.emb`) + edges (from `.edg`). Runs the
    /// same [`preprocess_topology`] the live embedder uses to derive `size_multipliers`
    /// and `edge_alphas`, so downstream code sees an identically-shaped `Topology`.
    pub fn new(positions: Vec<f32>, dim: usize, weights: Vec<f32>, edges: Vec<[u32; 2]>) -> Self {
        let (size_multipliers, edge_alphas) = preprocess_topology(&weights, &edges);
        Self {
            topology: Some(Topology {
                dim,
                edges,
                weights,
                size_multipliers,
                edge_alphas,
            }),
            positions,
            published: false,
        }
    }
}

impl TopologyProducer for LoadedEmbedding {
    fn poll(&mut self, positions_out: &mut Vec<f32>) -> Update {
        if self.published {
            return Update::Nothing;
        }
        self.published = true;
        positions_out.clear();
        positions_out.extend_from_slice(&self.positions);
        // Move the topology out — Update::Topology owns it after this. Trait defaults
        // for is_paused/is_finished/iteration/toggle_paused (all "already done") kick in
        // for subsequent polls.
        let Some(topo) = self.topology.take() else {
            // Shouldn't happen — we only take once. Defensive fallback keeps callers safe.
            return Update::Positions;
        };
        Update::Topology(topo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    /// Write `contents` to a temp file and return the handle (keeps it alive so the
    /// path stays valid) alongside its path. `Write::write_all`/`flush` resolve via the
    /// module-level `use std::io::Write` re-exported into scope by `use super::*`.
    fn temp_with(contents: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("create temp file");
        f.write_all(contents.as_bytes()).expect("write temp file");
        f.flush().expect("flush temp file");
        f
    }

    const EPS: f32 = 1e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    // ---- parse_embedding_file ----

    #[test]
    fn embedding_happy_path_sorts_and_interleaves() {
        // Ids out of order (2, 0, 1). Positions must land sorted-by-id, interleaved.
        let f = temp_with("2,20.0,21.0,2.5\n0,0.0,1.0,0.5\n1,10.0,11.0,1.5\n");
        let (positions, dim, weights) = parse_embedding_file(f.path()).expect("parse ok");
        assert_eq!(dim, 2);
        assert_eq!(positions, vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
        assert_eq!(weights, vec![0.5, 1.5, 2.5]);
    }

    #[test]
    fn embedding_skips_comments_and_blank_lines() {
        let f = temp_with("# header comment\n\n0,0.0,0.0,1.0\n   \n# mid comment\n1,2.0,3.0,4.0\n");
        let (positions, _, weights) = parse_embedding_file(f.path()).expect("parse ok");
        assert_eq!(positions, vec![0.0, 0.0, 2.0, 3.0]);
        assert_eq!(weights, vec![1.0, 4.0]);
    }

    #[test]
    fn embedding_duplicate_id_errors() {
        // ids 0, 0, 1 — after sort the contiguity check sees `0` where it expects `1`.
        let f = temp_with("0,0.0,0.0,1.0\n0,1.0,1.0,1.0\n1,2.0,2.0,1.0\n");
        let err = parse_embedding_file(f.path()).unwrap_err().to_string();
        assert!(err.contains("contiguous"), "unexpected error: {err}");
    }

    #[test]
    fn embedding_sparse_ids_errors_naming_gap() {
        // ids 0, 1, 3 — position 2 should be id 2, but sees id 3.
        let f = temp_with("0,0.0,0.0,1.0\n1,1.0,1.0,1.0\n3,3.0,3.0,1.0\n");
        let err = parse_embedding_file(f.path()).unwrap_err().to_string();
        assert!(err.contains("contiguous"), "unexpected error: {err}");
        assert!(
            err.contains("saw 3"),
            "error should name the gap value: {err}"
        );
        assert!(
            err.contains("position 2"),
            "error should name the gap position: {err}"
        );
    }

    #[test]
    fn embedding_high_dim_parses_with_id_sort() {
        // d=3, ids out of order — dim inference + the permutation-based coord reorder.
        let f = temp_with("1,10.0,11.0,12.0,1.5\n0,0.0,1.0,2.0,0.5\n");
        let (positions, dim, weights) = parse_embedding_file(f.path()).expect("parse ok");
        assert_eq!(dim, 3);
        assert_eq!(positions, vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
        assert_eq!(weights, vec![0.5, 1.5]);
    }

    #[test]
    fn embedding_1d_row_errors() {
        // id,x,w = 3 columns → d=1, below the viewer's minimum.
        let f = temp_with("0,0.0,1.0\n");
        let err = parse_embedding_file(f.path()).unwrap_err().to_string();
        assert!(err.contains("d >= 2"), "unexpected error: {err}");
    }

    #[test]
    fn embedding_mixed_column_counts_errors() {
        // First line 2D (4 cols), second line 3D (5 cols) → dim-changed error.
        let f = temp_with("0,0.0,0.0,1.0\n1,1.0,1.0,1.0,1.0\n");
        let err = parse_embedding_file(f.path()).unwrap_err().to_string();
        assert!(err.contains("dim changed"), "unexpected error: {err}");
    }

    #[test]
    fn embedding_non_numeric_x_errors_with_line_number() {
        let f = temp_with("0,0.0,0.0,1.0\n1,notanumber,1.0,1.0\n");
        let err = parse_embedding_file(f.path()).unwrap_err().to_string();
        assert!(
            err.contains("line 2"),
            "error should carry the line number: {err}"
        );
        assert!(
            err.contains("bad coord 1"),
            "error should name the bad field: {err}"
        );
    }

    #[test]
    fn embedding_empty_file_is_ok_with_zero_vertices() {
        // Pinned behavior: an empty (or all-comment) file parses to zero vertices.
        let f = temp_with("");
        let (positions, dim, weights) = parse_embedding_file(f.path()).expect("parse ok");
        assert_eq!(dim, 2);
        assert!(positions.is_empty());
        assert!(weights.is_empty());
    }

    // ---- parse_edges_file ----

    #[test]
    fn edges_happy_path_with_comments_and_blanks() {
        let f = temp_with("# comment\n0 1\n\n1 2\n  \n2 0\n");
        let edges = parse_edges_file(f.path(), 3).expect("parse ok");
        assert_eq!(edges, vec![[0, 1], [1, 2], [2, 0]]);
    }

    #[test]
    fn edges_out_of_range_vertex_errors_naming_line_and_ids() {
        // n_vertices = 3, so id 5 is out of range on line 2.
        let f = temp_with("0 1\n1 5\n");
        let err = parse_edges_file(f.path(), 3).unwrap_err().to_string();
        assert!(
            err.contains("line 2"),
            "error should carry the line number: {err}"
        );
        assert!(err.contains("out of range"), "unexpected error: {err}");
        assert!(
            err.contains("dst=5"),
            "error should name the offending id: {err}"
        );
    }

    #[test]
    fn edges_missing_dst_errors() {
        let f = temp_with("0\n");
        let err = parse_edges_file(f.path(), 3).unwrap_err().to_string();
        assert!(
            err.contains("missing/invalid dst"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn edges_extra_trailing_tokens_ignored() {
        // Pinned behavior: only the first two whitespace tokens are read; the rest ignored.
        let f = temp_with("0 1 weight ignored extra\n");
        let edges = parse_edges_file(f.path(), 3).expect("parse ok");
        assert_eq!(edges, vec![[0, 1]]);
    }

    // ---- save_embedding_file round-trip ----

    #[test]
    fn save_then_parse_round_trips() {
        // d=3 exercises the general per-coord loop; d=2 shares the same code path.
        let positions = vec![0.0_f32, 1.0, -7.5, 2.5, -3.25, 100.125];
        let weights = vec![1.0_f32, 2.0];
        let out = NamedTempFile::new().expect("create temp file");
        save_embedding_file(out.path(), &positions, 3, &weights).expect("save ok");
        let (rp, rdim, rw) = parse_embedding_file(out.path()).expect("parse ok");
        assert_eq!(rdim, 3);
        assert_eq!(rp.len(), positions.len());
        for (a, b) in rp.iter().zip(positions.iter()) {
            assert!(close(*a, *b), "position round-trip {a} vs {b}");
        }
        assert_eq!(rw.len(), weights.len());
        for (a, b) in rw.iter().zip(weights.iter()) {
            assert!(close(*a, *b), "weight round-trip {a} vs {b}");
        }
    }

    #[test]
    fn save_weights_length_mismatch_errors() {
        // 3 vertices worth of positions but only 2 weights.
        let positions = vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0];
        let weights = vec![1.0_f32, 2.0];
        let out = NamedTempFile::new().expect("create temp file");
        let err = save_embedding_file(out.path(), &positions, 2, &weights)
            .unwrap_err()
            .to_string();
        assert!(err.contains("weights.len()"), "unexpected error: {err}");
    }
}
