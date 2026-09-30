//! Vertex picking for hover-tooltips.
//!
//! The Picker wraps an `sprk::Sprk<2>` k-d tree over a frozen point cloud and answers
//! "which vertex is under this cursor?" queries. It owns the pick-radius policy (slack
//! multiplier applied to the rendered point radius) and the tie-break rule (closest hit
//! wins), so the App doesn't need to know either.
//!
//! Lifecycle is caller-driven — the App builds a Picker on running→paused and drops it on
//! paused→running. Positions are frozen while paused (the worker skips `calculate_step`),
//! so a single tree serves the entire paused interval. No `update` / retraining API here
//! because there's no paused-time scenario that would need one.

use sprk::{IdDist, Sprk};

/// Slack applied to the rendered point radius when querying. Generous — hover should not
/// require pixel-perfect aim, especially on sparse layouts where the nearest dot is far
/// from the cursor. Overlapping hits are tiebroken by distance-to-cursor so the extra reach
/// never mis-identifies the target when the user *is* being precise.
const PICK_RADIUS_MULT: f32 = 2.5;

pub struct Picker {
    tree: Sprk<2>,
}

impl Picker {
    /// Build a Picker from a flat `[x0, y0, x1, y1, ...]` position buffer. `None` when the
    /// buffer is empty — the caller can then skip picking without an extra guard.
    pub fn for_positions(positions: &[f32]) -> Option<Self> {
        debug_assert!(
            positions.len().is_multiple_of(2),
            "positions must be xy-interleaved"
        );
        if positions.is_empty() {
            return None;
        }
        // `bytemuck::cast_slice` reinterprets `[f32]` as `[[f32; 2]]` without copying —
        // sprk wants the fixed-arity view for its const-generic dimension.
        let pts: &[[f32; 2]] = bytemuck::cast_slice(positions);
        Some(Self {
            tree: Sprk::<2>::new(pts),
        })
    }

    /// Return the vertex closest to `cursor_world` within `PICK_RADIUS_MULT * rendered_radius`,
    /// or `None` if no vertex is inside that radius. Called once per CursorMoved event while
    /// paused — must stay cheap.
    pub fn hit_test(&self, cursor_world: [f32; 2], rendered_radius: f32) -> Option<u32> {
        let radius = (rendered_radius * PICK_RADIUS_MULT).max(1e-6);
        let mut hits: Vec<IdDist<u32, f32>> = Vec::new();
        self.tree.query_radius(&cursor_world, radius, &mut hits);
        hits.iter()
            .min_by(|a, b| a.dist.partial_cmp(&b.dist).expect("NaN in pick distance"))
            .map(|h| h.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_positions_is_none() {
        assert!(Picker::for_positions(&[]).is_none());
    }

    #[test]
    fn hit_within_radius_returns_nearest_miss_returns_none() {
        // Two vertices far apart. Effective radius = rendered_radius * PICK_RADIUS_MULT.
        let positions = [0.0_f32, 0.0, 100.0, 100.0];
        let picker = Picker::for_positions(&positions).expect("some picker");

        // Cursor near vertex 0 with a small radius → hits 0.
        let hit = picker.hit_test([0.1, 0.0], 1.0);
        assert_eq!(hit, Some(0));

        // Cursor far from any vertex with a tiny radius → miss.
        let miss = picker.hit_test([50.0, 50.0], 0.1);
        assert_eq!(miss, None);
    }

    #[test]
    fn two_candidates_in_radius_closer_wins() {
        // Both vertices within the pick radius of the cursor; the nearer one wins.
        let positions = [0.0_f32, 0.0, 3.0, 0.0];
        let picker = Picker::for_positions(&positions).expect("some picker");
        // Cursor at x=1: distance to v0 is 1, to v1 is 2. Radius (rendered 5 * 2.5 = 12.5)
        // covers both → nearer v0 wins.
        assert_eq!(picker.hit_test([1.0, 0.0], 5.0), Some(0));
        // Cursor at x=2.5: distance to v1 is 0.5, to v0 is 2.5 → v1 wins.
        assert_eq!(picker.hit_test([2.5, 0.0], 5.0), Some(1));
    }
}
