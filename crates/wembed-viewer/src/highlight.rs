//! Fade-others highlight state.
//!
//! Two independent sources contribute vertices:
//! - `search`: vertices whose label contains the current query substring.
//! - `hover`: the hovered vertex plus its 1-hop neighbors.
//!
//! The mask uploaded to the GPU is the **union** of the two sets. When both are
//! empty the mask is all-1 and the fragment shader multiplies alpha by 1, so
//! the graph renders exactly as it does without any highlight system in the
//! loop.
//!
//! Adjacency is (re)built once per topology publish. Search runs against a
//! precomputed lowercase copy of the labels — one-time O(N * avg_label_len)
//! allocation, then O(N * label_len) per query which stays under 20ms even on
//! `snap_amazon_all` (322k labels, ~200 bytes avg).
//!
//! Debouncing lives in the caller (App): this module doesn't own an Instant
//! clock. `set_query` is called only after the caller has decided the user is
//! done typing.
//!
//! Mask elements are `u32` (not `u8`) because that's what WGSL storage arrays
//! want without a lane-repack in the shader. 4× memory over a bit array but at
//! these graph sizes it's still under 2MB.

/// Alpha multiplier for faded (non-highlighted) vertices. Kept in Rust just as
/// documentation; the number that actually drives the shader lives in
/// `shaders/points.wgsl` / `shaders/lines.wgsl` as a `mix(FADE, 1.0, mask)`.
#[allow(dead_code)]
pub const FADE_ALPHA: f32 = 0.05;

pub struct Highlight {
    /// One entry per vertex. `1` = bright, `0` = faded. Uploaded to the GPU on
    /// every `drain_dirty()` — the callback pattern lets App tie uploads to the
    /// wgpu queue without Highlight holding a device reference.
    mask: Vec<u32>,

    /// `adj[v]` is the vertex ids of v's 1-hop neighbors (both directions —
    /// input edges are already undirected, but we mirror to make hover
    /// expansion a single lookup).
    adj: Vec<Vec<u32>>,

    /// Lowercase copies of the label strings, indexed by vertex id. `None` when
    /// no labels are loaded or when the labels file length doesn't match the
    /// current topology — search silently no-ops in that case.
    labels_lower: Option<Vec<String>>,

    /// The query that produced the current `search_set`. Blank string means
    /// "search inactive."
    active_query: String,
    /// Vertex ids matched by the current active_query. Kept as `Vec` because
    /// we only ever iterate — never test membership.
    search_set: Vec<u32>,

    /// Currently hovered vertex, if any.
    hover: Option<u32>,
    /// The hovered vertex plus its 1-hop neighbors. Same Vec-not-Set rationale
    /// as `search_set`.
    hover_set: Vec<u32>,

    /// Set whenever `mask` has been rewritten since the last `drain_dirty`.
    dirty: bool,
}

impl Highlight {
    pub fn new() -> Self {
        Self {
            mask: Vec::new(),
            adj: Vec::new(),
            labels_lower: None,
            active_query: String::new(),
            search_set: Vec::new(),
            hover: None,
            hover_set: Vec::new(),
            dirty: false,
        }
    }

    /// Reset everything for a new topology. Called on the initial topology
    /// publish and on every layered-embedder layer transition.
    ///
    /// `labels` — the same slice App would pass to `Attributes::label_for` for
    /// this layer. `None` if no labels are loaded or the count doesn't match
    /// (matches `Attributes::colors_for`'s silent-fallback semantics).
    pub fn set_topology(
        &mut self,
        num_vertices: usize,
        edges: &[[u32; 2]],
        labels: Option<&[String]>,
    ) {
        let mut adj = vec![Vec::new(); num_vertices];
        for &[u, v] in edges {
            let (u, v) = (u as usize, v as usize);
            if u < num_vertices && v < num_vertices && u != v {
                adj[u].push(v as u32);
                adj[v].push(u as u32);
            }
        }
        self.adj = adj;
        self.labels_lower =
            labels.map(|ls| ls.iter().map(|s| s.to_lowercase()).collect::<Vec<_>>());

        // Reset active state — a new topology invalidates the vertex ids in
        // both sets.
        self.active_query.clear();
        self.search_set.clear();
        self.hover = None;
        self.hover_set.clear();

        // All-bright by default. `max(1)` avoids a zero-sized storage buffer.
        self.mask = vec![1u32; num_vertices.max(1)];
        self.dirty = true;
    }

    /// Apply a new search query and refresh the mask. Idempotent — passing the
    /// same query twice does nothing on the second call. Caller is responsible
    /// for debouncing typing.
    pub fn set_query(&mut self, query: &str) {
        if query == self.active_query {
            return;
        }
        self.active_query.clear();
        self.active_query.push_str(query);

        self.search_set.clear();
        if !self.active_query.is_empty() {
            if let Some(ls) = self.labels_lower.as_deref() {
                // Lowercase-once; `contains` on precomputed lowercase strings
                // is the whole substring algorithm — nothing fancier needed
                // for the sizes we see.
                let q = self.active_query.to_lowercase();
                for (i, label) in ls.iter().enumerate() {
                    if label.contains(&q) {
                        self.search_set.push(i as u32);
                    }
                }
            }
        }
        self.rebuild_mask();
    }

    /// Update the hover neighborhood. `None` clears it.
    pub fn set_hover(&mut self, hover: Option<u32>) {
        if self.hover == hover {
            return;
        }
        self.hover = hover;
        self.hover_set.clear();
        if let Some(h) = hover {
            let h_idx = h as usize;
            if h_idx < self.adj.len() {
                self.hover_set.push(h);
                self.hover_set.extend(self.adj[h_idx].iter().copied());
            }
        }
        self.rebuild_mask();
    }

    /// Return the mask if it changed since the last call, else `None`. Clears
    /// the dirty flag so subsequent calls with no state change return `None`.
    pub fn drain_dirty_mask(&mut self) -> Option<&[u32]> {
        if !self.dirty {
            return None;
        }
        self.dirty = false;
        Some(&self.mask)
    }

    /// Count of vertices matched by the current query. Used by the stats panel
    /// to give the user typing feedback ("47 matches") without leaking Highlight
    /// internals.
    pub fn search_match_count(&self) -> usize {
        self.search_set.len()
    }

    /// Whether search is even meaningful — false when `--labels` wasn't passed
    /// or the labels file length didn't match the current topology. Used by
    /// the UI to hint "(no labels loaded)" instead of leaving the user typing
    /// into a field that will never match anything.
    pub fn has_labels(&self) -> bool {
        self.labels_lower.is_some()
    }

    fn rebuild_mask(&mut self) {
        let has_search = !self.search_set.is_empty();
        let has_hover = !self.hover_set.is_empty();
        if !has_search && !has_hover {
            // No highlight active — clear the fade so the graph renders as
            // normal. Faster than iterating the two empty sets to mark nothing.
            for slot in &mut self.mask {
                *slot = 1;
            }
        } else {
            for slot in &mut self.mask {
                *slot = 0;
            }
            for &v in &self.search_set {
                if let Some(slot) = self.mask.get_mut(v as usize) {
                    *slot = 1;
                }
            }
            for &v in &self.hover_set {
                if let Some(slot) = self.mask.get_mut(v as usize) {
                    *slot = 1;
                }
            }
        }
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn fresh_topology_is_all_bright_and_dirty_once() {
        let mut h = Highlight::new();
        let edges = [[0u32, 1], [1, 2]];
        h.set_topology(3, &edges, None);
        // Drain once → all-bright mask.
        let mask = h.drain_dirty_mask().expect("dirty after set_topology");
        assert_eq!(mask, &[1u32, 1, 1]);
        // Draining again with no change → None.
        assert!(h.drain_dirty_mask().is_none());
    }

    #[test]
    fn search_substring_case_insensitive_fades_the_rest() {
        let mut h = Highlight::new();
        let ls = labels(&["Apple", "banana", "grApe"]);
        h.set_topology(3, &[], Some(&ls));
        h.drain_dirty_mask(); // clear initial dirty

        // "ap" matches "Apple" (0) and "grApe" (2) case-insensitively; "banana" (1) fades.
        h.set_query("AP");
        let mask = h.drain_dirty_mask().expect("dirty after query");
        assert_eq!(mask, &[1u32, 0, 1]);
        assert_eq!(h.search_match_count(), 2);
    }

    #[test]
    fn query_with_no_labels_is_noop() {
        let mut h = Highlight::new();
        h.set_topology(3, &[], None);
        h.drain_dirty_mask();
        assert!(!h.has_labels());

        h.set_query("anything");
        // No labels → search set stays empty → mask stays all-bright. rebuild_mask still
        // marks dirty, but the mask is unchanged (all 1s) and no matches.
        assert_eq!(h.search_match_count(), 0);
        if let Some(mask) = h.drain_dirty_mask() {
            assert_eq!(mask, &[1u32, 1, 1]);
        }
    }

    #[test]
    fn hover_brightens_vertex_and_one_hop_neighbors() {
        let mut h = Highlight::new();
        // Star: 0 connected to 1 and 2; 3 is isolated.
        let edges = [[0u32, 1], [0, 2]];
        h.set_topology(4, &edges, None);
        h.drain_dirty_mask();

        h.set_hover(Some(0));
        let mask = h.drain_dirty_mask().expect("dirty after hover");
        // 0, 1, 2 bright; 3 faded.
        assert_eq!(mask, &[1u32, 1, 1, 0]);
    }

    #[test]
    fn hover_and_search_union_both_bright() {
        let mut h = Highlight::new();
        let ls = labels(&["zero", "one", "two", "three"]);
        // 0—1 edge; 2 and 3 isolated.
        let edges = [[0u32, 1]];
        h.set_topology(4, &edges, Some(&ls));
        h.drain_dirty_mask();

        h.set_query("three"); // matches vertex 3
        h.set_hover(Some(0)); // brightens 0 and its neighbor 1
        let mask = h.drain_dirty_mask().expect("dirty after union");
        // 0,1 (hover) + 3 (search) bright; 2 faded.
        assert_eq!(mask, &[1u32, 1, 0, 1]);
    }

    #[test]
    fn clearing_both_returns_all_bright() {
        let mut h = Highlight::new();
        let ls = labels(&["a", "b", "c"]);
        h.set_topology(3, &[[0u32, 1]], Some(&ls));
        h.drain_dirty_mask();

        h.set_query("a");
        h.set_hover(Some(1));
        h.drain_dirty_mask();

        h.set_query("");
        h.set_hover(None);
        let mask = h.drain_dirty_mask().expect("dirty after clearing");
        assert_eq!(mask, &[1u32, 1, 1]);
    }

    #[test]
    fn idempotent_same_query_and_hover_not_dirty_second_time() {
        let mut h = Highlight::new();
        let ls = labels(&["alpha", "beta"]);
        h.set_topology(2, &[[0u32, 1]], Some(&ls));
        h.drain_dirty_mask();

        h.set_query("alpha");
        assert!(h.drain_dirty_mask().is_some(), "first query is dirty");
        h.set_query("alpha");
        assert!(h.drain_dirty_mask().is_none(), "same query again not dirty");

        h.set_hover(Some(0));
        assert!(h.drain_dirty_mask().is_some(), "first hover is dirty");
        h.set_hover(Some(0));
        assert!(h.drain_dirty_mask().is_none(), "same hover again not dirty");
    }

    #[test]
    fn self_loops_and_out_of_range_edges_no_panic() {
        let mut h = Highlight::new();
        // Self-loop on 0, out-of-range edge (5 >= 3), and a valid edge.
        let edges = [[0u32, 0], [1, 5], [0, 1]];
        h.set_topology(3, &edges, None);
        h.drain_dirty_mask();

        // Hover on 0: neighbors from adjacency (self-loop skipped, out-of-range skipped).
        h.set_hover(Some(0));
        let mask = h.drain_dirty_mask().expect("dirty");
        // 0 and its valid neighbor 1 bright; 2 faded.
        assert_eq!(mask, &[1u32, 1, 0]);
    }
}
