//! Per-vertex attributes: file parsers plus a runtime bag that owns the silent-fallback
//! rules ("colors apply only when file length matches the current N", "label if in-range,
//! else `#<vertex_id>`"). Keeping those rules on `Attributes` rather than the App means
//! adding a new per-vertex attribute later (size, edge color, ...) is one file to touch.
//!
//! Both files use line-index = vertex-id. Parsed eagerly at CLI-parse time so typos fail
//! before the (potentially slow) graph load kicks off. Colors return as packed RGBA8 u32s
//! ready to upload straight into the WGSL storage buffer and unpack via `unpack4x8unorm`.

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::cli::VertexColorSource;
use crate::theme::{pack_rgba, Theme};

/// Parse a colors file: one hex color per line. `#` prefix is optional, alpha channel is
/// optional (defaults to `ff`), digits are case-insensitive. Empty lines and lines with
/// wrong digit count are hard errors — colors are meant to be machine-generated, so any
/// deviation is a caller bug worth surfacing.
///
/// Returned `u32`s are packed as little-endian `[R, G, B, A]` bytes so WGSL's
/// `unpack4x8unorm(v).xyzw` reads out `(R, G, B, A)` directly.
pub fn parse_colors_file(path: &Path) -> Result<Vec<u32>> {
    let file = File::open(path).with_context(|| format!("open colors file {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let lineno = idx + 1;
        let raw = line.with_context(|| format!("read colors line {lineno}"))?;
        let s = raw.trim();
        let s = s.strip_prefix('#').unwrap_or(s);
        let (r, g, b, a) = match s.len() {
            6 => {
                let v = u32::from_str_radix(s, 16)
                    .with_context(|| format!("colors line {lineno}: invalid hex `{s}`"))?;
                (
                    ((v >> 16) & 0xff) as u8,
                    ((v >> 8) & 0xff) as u8,
                    (v & 0xff) as u8,
                    0xff,
                )
            }
            8 => {
                let v = u32::from_str_radix(s, 16)
                    .with_context(|| format!("colors line {lineno}: invalid hex `{s}`"))?;
                (
                    ((v >> 24) & 0xff) as u8,
                    ((v >> 16) & 0xff) as u8,
                    ((v >> 8) & 0xff) as u8,
                    (v & 0xff) as u8,
                )
            }
            _ => bail!("colors line {lineno}: expected 6 or 8 hex digits, got `{s}`"),
        };
        out.push(u32::from_le_bytes([r, g, b, a]));
    }
    Ok(out)
}

/// Parse a categories file: one integer per line, line i == vertex i's category id.
/// `-1` (or any negative) marks the vertex as uncategorized — the theme's neutral base04
/// color is used instead of an accent slot. Blank lines and non-integers are hard
/// errors — categories are machine-generated.
pub fn parse_categories_file(path: &Path) -> Result<Vec<i32>> {
    let file =
        File::open(path).with_context(|| format!("open categories file {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let lineno = idx + 1;
        let raw = line.with_context(|| format!("read categories line {lineno}"))?;
        let s = raw.trim();
        if s.is_empty() {
            bail!("categories line {lineno}: empty (expected integer)");
        }
        let v: i32 = s
            .parse()
            .with_context(|| format!("categories line {lineno}: invalid integer `{s}`"))?;
        out.push(v);
    }
    Ok(out)
}

/// Parse a category-names file: plain text, line i == the display name of category id i
/// (the `--categories` id space). Lines are trimmed; blank lines are KEPT as empty
/// strings so the ids below them don't shift — the legend builder skips them.
pub fn parse_category_names_file(path: &Path) -> Result<Vec<String>> {
    let file =
        File::open(path).with_context(|| format!("open category-names file {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let raw = line.with_context(|| format!("read category-names line {}", idx + 1))?;
        out.push(raw.trim().to_string());
    }
    Ok(out)
}

/// Parse a labels file: one JSON string per line (JSON Lines format). Standard JSON
/// escaping covers newlines (`"foo\nbar"`), unicode, and quotes with zero custom parser
/// surface area.
pub fn parse_labels_file(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("open labels file {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let lineno = idx + 1;
        let raw = line.with_context(|| format!("read labels line {lineno}"))?;
        let label: String = serde_json::from_str(raw.trim())
            .with_context(|| format!("labels line {lineno}: invalid JSON string"))?;
        out.push(label);
    }
    Ok(out)
}

/// Runtime bag of per-vertex attributes loaded from CLI files. Owns the silent-fallback
/// rules so callers can query with just a vertex count / vertex id and get a well-defined
/// answer without duplicating "did the file match?" checks.
///
/// Grows as more attributes land — per-vertex size, per-edge color, etc. each become a
/// private field plus one accessor with its own fallback rule.
pub struct Attributes {
    colors: Option<Vec<u32>>,
    categories: Option<Vec<i32>>,
    labels: Option<Vec<String>>,
    /// Per-edge packed RGBA8 colors from `--edge-colors` (line i = edge i). Consumed by the
    /// `file` edge-color source; applied only when the length matches the edge count.
    edge_colors: Option<Vec<u32>>,
    /// Cached resolution of `theme × categories → packed RGBA per vertex`. Rebuilt lazily
    /// via `RefCell` on the first read after either input changes. Keeps `colors_for`'s
    /// `&self` signature intact so App call sites don't change.
    resolved_theme_colors: RefCell<Option<Vec<u32>>>,
    /// Snapshot of the theme name (+ its accents length) at the time `resolved_theme_colors`
    /// was built. Used to detect "theme changed" and invalidate the cache.
    resolved_theme_key: RefCell<Option<(String, usize, u32)>>,
}

impl Attributes {
    pub fn new(
        colors: Option<Vec<u32>>,
        categories: Option<Vec<i32>>,
        labels: Option<Vec<String>>,
        edge_colors: Option<Vec<u32>>,
    ) -> Self {
        Self {
            colors,
            categories,
            labels,
            edge_colors,
            resolved_theme_colors: RefCell::new(None),
            resolved_theme_key: RefCell::new(None),
        }
    }

    /// Per-vertex colors for a scene of `n` vertices under `source`: the `--colors` file, or
    /// the theme accents indexed by `--categories` (`-1` → the theme's neutral base04).
    /// `None` — the shader then draws the theme's vertex color — for `theme`, when the
    /// selected file isn't loaded, or when its line count doesn't match `n` (a layered run's
    /// coarser layers show the theme color, the finest layer lights up).
    pub fn colors_for(
        &self,
        source: VertexColorSource,
        n: usize,
        theme: Option<&Theme>,
        uncategorized_alpha: f32,
    ) -> Option<Vec<u32>> {
        match source {
            VertexColorSource::Theme => None,
            VertexColorSource::File => self
                .colors
                .as_deref()
                .filter(|cs| cs.len() == n)
                .map(<[u32]>::to_vec),
            VertexColorSource::Categories => self.category_colors(n, theme?, uncategorized_alpha),
        }
    }

    pub fn has_colors(&self) -> bool {
        self.colors.is_some()
    }

    pub fn has_categories(&self) -> bool {
        self.categories.is_some()
    }

    pub fn has_edge_colors(&self) -> bool {
        self.edge_colors.is_some()
    }

    fn category_colors(
        &self,
        n: usize,
        theme: &Theme,
        uncategorized_alpha: f32,
    ) -> Option<Vec<u32>> {
        let cats = self.categories.as_deref()?;
        if cats.len() != n {
            return None;
        }
        // Cache hit: theme + accent count unchanged AND cached buffer sized right.
        let cache_key = (
            theme.name.clone(),
            theme.accents().len(),
            uncategorized_alpha.to_bits(),
        );
        if self.resolved_theme_key.borrow().as_ref() == Some(&cache_key) {
            if let Some(cached) = self.resolved_theme_colors.borrow().as_ref() {
                if cached.len() == n {
                    return Some(cached.clone());
                }
            }
        }
        // Miss: resolve fresh.
        let accents = theme.accents();
        let mut neutral_rgba = theme.neutral();
        neutral_rgba[3] =
            (neutral_rgba[3] as f32 * uncategorized_alpha.clamp(0.0, 1.0)).round() as u8;
        let neutral = pack_rgba(neutral_rgba);
        let mut out = Vec::with_capacity(n);
        for &c in cats {
            if c < 0 {
                out.push(neutral);
            } else {
                let idx = (c as usize) % accents.len();
                out.push(pack_rgba(accents[idx]));
            }
        }
        *self.resolved_theme_colors.borrow_mut() = Some(out.clone());
        *self.resolved_theme_key.borrow_mut() = Some(cache_key);
        Some(out)
    }

    /// Invalidate the theme-resolved-colors cache. Call whenever categories load/change,
    /// or whenever the caller expects to switch themes soon — cheap.
    pub fn invalidate_theme_cache(&self) {
        *self.resolved_theme_colors.borrow_mut() = None;
        *self.resolved_theme_key.borrow_mut() = None;
    }

    /// Text to show for a hovered vertex: the label when a labels file was loaded, its
    /// length matches the current topology's vertex count `n`, AND the id is in-range;
    /// otherwise `#<id>`. Always returns something — the caller never needs a "no label"
    /// branch.
    ///
    /// The length check matters during layered-embedder runs: coarser layers have fewer
    /// vertices than the labels file was written for, and vertex ids in those layers refer
    /// to coarsened supernodes rather than the products the labels describe. Silently
    /// falling back to `#<id>` matches the same rule `colors_for` uses.
    pub fn label_for(&self, vertex: u32, n: usize) -> String {
        self.labels
            .as_deref()
            .filter(|ls| ls.len() == n)
            .and_then(|ls| ls.get(vertex as usize))
            .cloned()
            .unwrap_or_else(|| format!("#{vertex}"))
    }

    /// Labels slice for a scene of `n` vertices, using the same silent-fallback rule as
    /// `colors_for`: `Some` only when a file was loaded AND the length matches. Used by
    /// `Highlight` to build a searchable index — the length match matters because
    /// mid-embed layered transitions can shrink N, invalidating the label indices.
    pub fn labels_for(&self, n: usize) -> Option<&[String]> {
        match self.labels.as_deref() {
            Some(ls) if ls.len() == n => Some(ls),
            _ => None,
        }
    }

    /// Per-edge packed RGBA8 colors for a graph of `n_edges` edges, using the same
    /// silent-fallback rule as `labels_for`: `Some` only when a `--edge-colors` file was
    /// loaded AND its line count equals the edge count. On a mismatch (a layered coarser
    /// layer, or the wrong file) the `file` edge-color source is downgraded to `theme`
    /// upstream, so `None` here is a well-defined "no per-edge colors."
    pub fn edge_colors_for(&self, n_edges: usize) -> Option<&[u32]> {
        match self.edge_colors.as_deref() {
            Some(cs) if cs.len() == n_edges => Some(cs),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::NamedTempFile;

    fn temp_with(contents: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("create temp file");
        f.write_all(contents.as_bytes()).expect("write temp file");
        f.flush().expect("flush temp file");
        f
    }

    /// Minimal base16 YAML with all 16 base slots present and distinct accents.
    fn base16_yaml(name: &str) -> String {
        let mut s = format!("system: \"base16\"\nname: \"{name}\"\npalette:\n");
        for i in 0..16u32 {
            // Make each slot a distinct color derived from its index.
            let byte = (i * 16) as u8;
            s.push_str(&format!(
                "  base{i:02X}: \"#{byte:02x}{byte:02x}{byte:02x}\"\n"
            ));
        }
        s
    }

    // ---- parse_colors_file ----

    #[test]
    fn colors_6_digit_defaults_alpha_ff_little_endian() {
        let f = temp_with("112233\n");
        let out = parse_colors_file(f.path()).expect("parse ok");
        // Packed little-endian [R,G,B,A] = [0x11, 0x22, 0x33, 0xff].
        assert_eq!(out, vec![u32::from_le_bytes([0x11, 0x22, 0x33, 0xff])]);
    }

    #[test]
    fn colors_8_digit_preserves_alpha() {
        let f = temp_with("11223344\n");
        let out = parse_colors_file(f.path()).expect("parse ok");
        assert_eq!(out, vec![u32::from_le_bytes([0x11, 0x22, 0x33, 0x44])]);
    }

    #[test]
    fn colors_optional_hash_and_case_insensitive() {
        let f = temp_with("#AaBbCc\nddEEff\n");
        let out = parse_colors_file(f.path()).expect("parse ok");
        assert_eq!(
            out,
            vec![
                u32::from_le_bytes([0xaa, 0xbb, 0xcc, 0xff]),
                u32::from_le_bytes([0xdd, 0xee, 0xff, 0xff]),
            ]
        );
    }

    #[test]
    fn colors_wrong_digit_count_errors_with_line_number() {
        let f = temp_with("112233\nxyz\n");
        let err = parse_colors_file(f.path()).unwrap_err().to_string();
        assert!(err.contains("line 2"), "unexpected error: {err}");
    }

    #[test]
    fn colors_invalid_hex_errors_with_line_number() {
        // Right length (6) but non-hex digits.
        let f = temp_with("GGHHII\n");
        let err = parse_colors_file(f.path()).unwrap_err().to_string();
        assert!(err.contains("line 1"), "unexpected error: {err}");
        assert!(err.contains("invalid hex"), "unexpected error: {err}");
    }

    // ---- parse_categories_file ----

    #[test]
    fn categories_parse_including_negatives() {
        let f = temp_with("0\n-1\n5\n-42\n");
        let out = parse_categories_file(f.path()).expect("parse ok");
        assert_eq!(out, vec![0, -1, 5, -42]);
    }

    #[test]
    fn categories_empty_line_errors() {
        let f = temp_with("0\n\n1\n");
        let err = parse_categories_file(f.path()).unwrap_err().to_string();
        assert!(err.contains("line 2"), "unexpected error: {err}");
        assert!(err.contains("empty"), "unexpected error: {err}");
    }

    #[test]
    fn categories_non_integer_errors() {
        let f = temp_with("0\n1.5\n");
        let err = parse_categories_file(f.path()).unwrap_err().to_string();
        assert!(err.contains("line 2"), "unexpected error: {err}");
        assert!(err.contains("invalid integer"), "unexpected error: {err}");
    }

    // ---- parse_labels_file ----

    #[test]
    fn labels_json_escapes_decode() {
        // Newline escape, unicode escape (é), and an escaped quote.
        let f = temp_with("\"foo\\nbar\"\n\"caf\\u00e9\"\n\"a \\\"quoted\\\" word\"\n");
        let out = parse_labels_file(f.path()).expect("parse ok");
        assert_eq!(out[0], "foo\nbar");
        assert_eq!(out[1], "café");
        assert_eq!(out[2], "a \"quoted\" word");
    }

    #[test]
    fn labels_non_string_json_errors() {
        let f_num = temp_with("42\n");
        assert!(parse_labels_file(f_num.path()).is_err());
        let f_obj = temp_with("{\"k\": 1}\n");
        assert!(parse_labels_file(f_obj.path()).is_err());
    }

    // ---- Attributes::colors_for ----

    #[test]
    fn uncategorized_alpha_dims_only_negative_categories() {
        // -1 vertices get the dimmed neutral; categorized vertices keep full accent
        // alpha; the alpha is part of the cache key so both variants coexist.
        let theme = Theme::from_yaml(&base16_yaml("A")).unwrap();
        let attrs = Attributes::new(None, Some(vec![0, -1, 1]), None, None);
        let full = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme), 1.0)
            .expect("some");
        let dim = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme), 0.5)
            .expect("some");
        assert_eq!(full[0], dim[0], "categorized vertex untouched");
        assert_eq!(full[2], dim[2], "categorized vertex untouched");
        assert_ne!(full[1], dim[1], "uncategorized vertex dimmed");
        // alpha byte halved (pack order verified elsewhere; compare via re-resolve)
        let full_again = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme), 1.0)
            .expect("some");
        assert_eq!(full, full_again, "cache key distinguishes alphas");
    }

    #[test]
    fn colors_for_follows_the_selected_source_without_fallthrough() {
        // With both files loaded each source reads its own; a length mismatch yields None
        // (theme color) rather than silently showing the other file.
        let direct = vec![10u32, 20, 30];
        let attrs = Attributes::new(Some(direct.clone()), Some(vec![0, 1, 2]), None, None);
        let theme = Theme::from_yaml(&base16_yaml("A")).unwrap();
        let accents = theme.accents();
        let file = attrs.colors_for(VertexColorSource::File, 3, Some(&theme), 1.0);
        assert_eq!(file, Some(direct));
        let cats = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme), 1.0)
            .expect("some");
        assert_eq!(cats[2], pack_rgba(accents[2]));
        assert!(attrs
            .colors_for(VertexColorSource::Theme, 3, Some(&theme), 1.0)
            .is_none());
        let short = Attributes::new(Some(vec![10u32, 20]), Some(vec![0, 1, 2]), None, None);
        assert!(short
            .colors_for(VertexColorSource::File, 3, Some(&theme), 1.0)
            .is_none());
    }

    #[test]
    fn colors_for_categories_length_mismatch_is_none() {
        let cats = vec![0i32, 1]; // length 2, but n = 3
        let attrs = Attributes::new(None, Some(cats), None, None);
        let theme = Theme::from_yaml(&base16_yaml("A")).unwrap();
        assert!(attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme), 1.0)
            .is_none());
    }

    #[test]
    fn colors_for_negative_is_neutral_and_wraps_via_modulo() {
        let theme = Theme::from_yaml(&base16_yaml("A")).unwrap();
        let accents = theme.accents();
        let n_acc = accents.len(); // 8 for base16
                                   // category -1 → neutral; category == n_acc → wraps to accents[0].
        let cats = vec![-1i32, 0, n_acc as i32];
        let attrs = Attributes::new(None, Some(cats), None, None);
        let got = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme), 1.0)
            .expect("some");
        assert_eq!(got[0], pack_rgba(theme.neutral()), "-1 → neutral");
        assert_eq!(got[1], pack_rgba(accents[0]), "0 → accents[0]");
        assert_eq!(got[2], pack_rgba(accents[0]), "n_acc wraps to accents[0]");
    }

    #[test]
    fn colors_for_cache_invalidated_by_theme_change_and_invalidate() {
        let cats = vec![0i32, 1, 2];
        let attrs = Attributes::new(None, Some(cats), None, None);
        let theme_a = Theme::from_yaml(&base16_yaml("Alpha")).unwrap();
        let theme_b = Theme::from_yaml(&base16_yaml("Beta")).unwrap();

        // First resolve populates the cache for theme A.
        let a1 = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme_a), 1.0)
            .expect("some");
        // Second resolve with A hits the cache and returns the same thing.
        let a2 = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme_a), 1.0)
            .expect("some");
        assert_eq!(a1, a2);

        // Switching to a differently-named theme must NOT return A's cached colors — the
        // cache key includes the theme name, so B re-resolves. (Same palette here, so the
        // values match, but the cache-key change is what we're exercising.)
        let b = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme_b), 1.0)
            .expect("some");
        assert_eq!(b.len(), 3);

        // After explicit invalidation the next resolve still returns correct colors.
        attrs.invalidate_theme_cache();
        let a3 = attrs
            .colors_for(VertexColorSource::Categories, 3, Some(&theme_a), 1.0)
            .expect("some");
        assert_eq!(a3, a1, "post-invalidate resolve should match original");
    }

    // ---- Attributes::label_for / labels_for ----

    #[test]
    fn label_for_returns_label_when_lengths_match_and_in_range() {
        let labels = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let attrs = Attributes::new(None, None, Some(labels), None);
        assert_eq!(attrs.label_for(1, 3), "beta");
        // Out of range id still falls back to `#<id>`.
        assert_eq!(attrs.label_for(9, 3), "#9");
    }

    #[test]
    fn edge_colors_for_applies_only_on_exact_edge_count_match() {
        let ecs = vec![1u32, 2, 3];
        let attrs = Attributes::new(None, None, None, Some(ecs.clone()));
        assert_eq!(attrs.edge_colors_for(3), Some(ecs.as_slice()));
        // Wrong edge count (a layered coarser layer) → None → downgrade to theme upstream.
        assert!(attrs.edge_colors_for(2).is_none());
        // No file loaded → None.
        let none = Attributes::new(None, None, None, None);
        assert!(none.edge_colors_for(3).is_none());
    }

    #[test]
    fn label_for_length_mismatch_falls_back_to_hash_id() {
        let labels = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let attrs = Attributes::new(None, None, Some(labels), None);
        // n=2 (coarse layer) doesn't match the 3-label file → `#<id>`.
        assert_eq!(attrs.label_for(0, 2), "#0");
        assert!(attrs.labels_for(2).is_none());
        assert!(attrs.labels_for(3).is_some());
    }
}
