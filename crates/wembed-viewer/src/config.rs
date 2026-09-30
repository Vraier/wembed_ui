//! TOML config file — a saveable, loadable snapshot of every runtime knob the viewer
//! exposes on the command line plus the interactive UI (visual channels, sizing, camera,
//! export resolution). Mirrors the CLI flags in [`crate::cli`] and the interactive
//! [`crate::app::VisualState`] one-to-one so a session can be persisted to disk and
//! replayed later.
//!
//! # Precedence contract
//!
//! The intended layering is **defaults → TOML → explicitly-passed CLI flags**: a config
//! file overrides the compiled-in defaults, and a CLI flag the user actually typed
//! overrides the config file. That resolution is the *consumer's* job — this module only
//! parses the file into a fully-optional [`FileConfig`]. Every field is an `Option<T>`
//! precisely so the consumer can tell "the file set this" apart from "the file left it
//! alone" and merge accordingly. Nothing here reads clap or touches defaults.
//!
//! # Path resolution
//!
//! Path-valued fields (`graph`, `embedding`, `attrs.colors`/`labels`/`categories`,
//! `theme.dir`) are resolved *relative to the directory containing the TOML file* by
//! [`FileConfig::load`], so a config can be moved around with its data as a bundle.
//! Absolute paths pass through untouched. This mirrors how most tools treat paths in a
//! config file (relative-to-the-config, not relative-to-CWD).
//!
//! # `[camera]` applicability
//!
//! A saved camera (`center` + `half` + `rotation_deg`) only makes sense against a *fixed*
//! embedding — a live embedder relays out every step, so a frozen viewport would fight the
//! auto-refit. The consumer is expected to apply the *entire* `[camera]` section (rotation
//! included) only when an `.emb` producer is active (i.e. `embedding` is set); this module
//! just carries the values and enforces that `center` and `half` travel together
//! (`rotation_deg` is independently optional — see [`FileConfig::load`]).

// The consumers (CLI/UI merge, camera application) land in a follow-up change; until then
// most of this module is legitimately unreferenced. Silence the noise rather than
// scatter per-item `#[allow]`s.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::cli::{EdgeColorSource, LrSchedule, ScalarSource, StopCriterion, VertexColorSource};
use crate::projection::ProjectionMode;

/// Embedder knobs — the forwarded `wembed::Options` subset (see [`crate::cli::Cli`]).
/// Ignored by the consumer when an `.emb` `embedding` is loaded (the embedder is skipped
/// entirely in that case), same as the CLI flags. Field names mirror the TOML keys; the
/// CLI flags `--speed`/`--lr-cooling` map to the `learning_rate`/`lr_cooling_factor` keys
/// here (the keys keep their descriptive names).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbedderConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    /// Embedding dimension (`--dimension`, >= 2). The display projects to 2D via
    /// `[visual] projection`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<i32>,
    /// `--dim-hint`: weights become degree^(dimension/dimension_hint); non-positive = off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension_hint: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_cooling_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layered: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centre: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dyn_buffer: Option<f64>,
    // ---- Learning-rate schedule (mirrors the new CLI flags) ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_schedule: Option<LrSchedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_steps: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_decay_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_decay_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_adapt_patience: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_growth_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_growth_threshold: Option<f64>,
    // ---- Stopping criterion (mirrors the new CLI flags) ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_criterion: Option<StopCriterion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_displacement_tol: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_displacement_patience: Option<i32>,
    // ---- Loss stop ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_loss_tol: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_loss_patience: Option<i32>,
}

/// Per-vertex attribute files — the `--colors` / `--labels` / `--categories` trio. Paths
/// are resolved relative to the config file's directory by [`FileConfig::load`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttrsConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub colors: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categories: Option<PathBuf>,
    /// Per-edge colors file (`--edge-colors`) — line i = edge i of the loaded `.edg`, same
    /// hex format as `colors`. Feeds the `file` edge-color source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_colors: Option<PathBuf>,
}

/// Theme selection — the `--themes-dir` / `--theme` pair. `name` is a case- and
/// separator-insensitive substring match against loaded theme names or file stems,
/// same rule as `--theme`. `dir` is resolved relative to the config file's directory.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThemeConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Visual channels — mirrors the color/size/alpha slider subset of
/// [`crate::app::VisualState`] plus the initial label-`search` query to re-apply on load.
/// Rotation lives in `[camera]` (it's a viewport property, only replayed for `.emb`).
/// Pure presentation state, no paths.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisualConfig {
    /// `theme` / `categories` / `file`. Absent → `theme`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertex_color_source: Option<VertexColorSource>,
    /// `constant` / `weight`. Absent → `weight`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertex_size_source: Option<ScalarSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertex_size_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertex_alpha_scale: Option<f32>,
    /// Alpha multiplier for uncategorized (`-1`) vertices under `--categories`
    /// coloring. `1.0` = no dimming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncategorized_alpha: Option<f32>,
    /// Per-edge color source (`theme` / `vertex` / `file` / `length`). Absent → `theme`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_color_source: Option<EdgeColorSource>,
    /// `constant` / `weight`. Absent → `weight`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_alpha_source: Option<ScalarSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_alpha_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_width_scale: Option<f32>,
    /// Stress-ratio clamp `[lo, hi]` for the `length` source's diverging colormap (r=1 is
    /// the neutral midpoint). Absent → automatic q05/q95 quantile window; setting a range
    /// fixes it manually.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_length_range: Option<[f32; 2]>,
    /// How dimension-> 2 embeddings are projected into the 2D view (`first-two` / `pca`).
    /// Ignored at dimension 2. Absent → `pca`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection: Option<ProjectionMode>,
    /// Strength (0..=1) of the depth shading — vertices fade by their out-of-projection-
    /// plane residual (quantile-normalized). `0` disables; only meaningful at dimension > 2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_fade: Option<f32>,
    /// Mirror the rendered image left↔right / top↔bottom. Lives here (not `[camera]`)
    /// because a mirror is meaningful for every producer, live embedder included — it
    /// composes with the auto-refit, unlike a frozen center/half.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flip_x: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flip_y: Option<bool>,
    /// Label-search query to re-apply on load (drives the highlight mask). Silently no-ops
    /// when no `--labels` are loaded, same as typing into the UI search box.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<String>,
}

/// Point-sizing knobs — the density-driven world target plus the pixel clamps
/// (`--density-factor` / `--point-min-pixels` / `--point-max-pixels`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SizingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density_factor: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_min_pixels: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_max_pixels: Option<f32>,
}

/// A frozen viewport — `center` (world-space), `half` (half-extent), and `rotation_deg`
/// (CCW degrees, mirrors [`crate::app::VisualState::rotation_deg`]). `center` and `half`
/// must be present together — [`FileConfig::load`] rejects a `[camera]` that sets only one
/// of the pair — but `rotation_deg` is independently optional, so a `[camera]` carrying
/// only a rotation is valid (the consumer auto-fits center/half and just applies the
/// rotation). The whole section is applied only when an `.emb` producer is active (see
/// module docs).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CameraConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center: Option<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub half: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation_deg: Option<f32>,
}

/// Density (heatmap) render mode. PRESENCE of this section enables the mode — it's
/// orthogonal to `[visual] vertex_color_source`, which keeps controlling the normal points/edges path
/// (ignored while density is on). The pipeline is accumulate → blur → normalize+colormap:
/// primitives splat their scalar weight into an off-screen buffer, a separable gaussian
/// smooths it, and a log tone-curve + OKLab ramp map density to color.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DensityConfig {
    /// Blur sigma in PIXELS (post-accumulation gaussian ≡ per-splat kernel by linearity).
    /// `0` bypasses both blur passes. Default 1.2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f32>,
    /// Tone curve. Only `"log"` for now; a hard error on anything else.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curve: Option<String>,
    /// Knee anchor as a percentile over LIT pixels (density > 0). Default 50.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knee_percentile: Option<f32>,
    /// White anchor as a percentile over LIT pixels. Default 99.5.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub white_percentile: Option<f32>,
    /// Explicit knee density value — overrides `knee_percentile` when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knee: Option<f32>,
    /// Explicit white density value — overrides `white_percentile` when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub white: Option<f32>,
    /// Colormap ramp — >= 2 hex stops, dark → bright, equal-t spacing, interpolated in
    /// OKLab. Absent → theme background → theme foreground.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ramp: Option<Vec<String>>,
}

/// Export resolution — the headless / save-button target dimensions in pixels.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExportConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

/// Title text drawn into the render (window + export, same quad).
/// The namespace is designed to grow (`[[text.labels]]` for in-scene labels later).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    /// The title string. Absent/empty = no text drawn. `\n` makes multiple lines.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Text height in px at the 1080p reference height — scales with the target like the
    /// point-size pixel clamps. Default: 28.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    /// Corner the title anchors to. Default: top-left.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<crate::text::Anchor>,
    /// Margins from the anchored corner, in 1080p-reference px (same scaling rule
    /// as `size`). Default: half the scaled text size. Useful to keep overlays
    /// inside the crop-safe zone when one wallpaper serves differing aspects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin_x: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin_y: Option<f32>,
    /// Hex text color (`#rrggbb` or `#rrggbbaa`). Default: the theme's foreground
    /// (base05). Validated at resolve time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Font file (.ttf/.otf) override. Relative → resolved against the config's
    /// directory. Default: the bundled JetBrains Mono Regular.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font: Option<PathBuf>,
}

/// One legend row: a color swatch + a label. The swatch color comes from EITHER an
/// explicit hex `color` OR a `category` id resolved through the active theme's accent
/// palette — the same `accents[category % len]` mapping the `--categories` file uses
/// (negative → the theme's neutral base04). Exactly one of the two must be set;
/// [`crate::settings`] rejects anything else at resolve time.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegendEntryConfig {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

/// Manual category legend drawn into the render (window + export, identical pixels —
/// same CPU-rasterized-quad machinery as `[text]`). Rows are `[[legend.entries]]` tables
/// in TOML order. No entries → no legend.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegendConfig {
    /// Row text height in px at the 1080p reference height (same scaling rule as
    /// `[text].size`). Default: 24.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    /// Corner the legend block anchors to. Default: bottom-left (the title defaults to
    /// top-left, so both fit without overlapping out of the box).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<crate::text::Anchor>,
    /// Margins from the anchored corner, in 1080p-reference px (same scaling rule
    /// as `size`). Default: half the scaled text size. Useful to keep overlays
    /// inside the crop-safe zone when one wallpaper serves differing aspects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin_x: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin_y: Option<f32>,
    /// Hex label-text color (`#rrggbb` / `#rrggbbaa`). Default: the theme's foreground.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Category-names file: plain text, line i = the display name of category id i
    /// (matching the `--categories` id space). Generates one entry per named category,
    /// colored by theme accent — an empty line skips that category without shifting the
    /// ids below it. Mutually exclusive with manual `entries` and `entries_from`.
    /// Relative → resolved against the config's directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub names: Option<PathBuf>,
    /// External TOML carrying only `[[entries]]` rows (same schema/validation as inline
    /// `entries`). Lets a generator that bakes a palette emit the matching legend once and
    /// share it across per-mode render configs. Mutually exclusive with `names` and inline
    /// `entries`. Relative → resolved against the config's directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entries_from: Option<PathBuf>,
    /// Rows, in order. Declared LAST: the toml serializer emits fields in struct order
    /// and rejects scalar values after a table array (`[[legend.entries]]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<LegendEntryConfig>,
}

/// External legend-entries document (`[legend].entries_from` / `--legend-entries`): a bare
/// list of `[[entries]]` rows with the same schema and validation as inline
/// `[[legend.entries]]`. Styling (anchor/size/color/font) stays in the main config.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegendEntriesFile {
    #[serde(default)]
    pub entries: Vec<LegendEntryConfig>,
}

impl LegendEntriesFile {
    pub fn load(path: &Path) -> Result<Vec<LegendEntryConfig>> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read legend entries file {}", path.display()))?;
        let doc: LegendEntriesFile = toml::from_str(&text)
            .with_context(|| format!("parse legend entries file {}", path.display()))?;
        Ok(doc.entries)
    }
}

/// Top-level config document. Every field is optional; a missing section deserializes to
/// `None` (the whole struct `#[derive(Default)]`s to all-`None` via [`FileConfig::empty`]).
/// `deny_unknown_fields` on every struct turns typos into errors instead of silent no-ops.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    /// Edge-list file (the positional `graph_path` on the CLI). Relative → resolved
    /// against the config's directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<PathBuf>,
    /// Precomputed `.emb` to load positions from (`--input-embedding`). Relative →
    /// resolved against the config's directory. Presence flips the consumer into "skip
    /// the embedder" mode; a missing file is a hard error, never a silent live embed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_embedding: Option<PathBuf>,
    /// Where `headless` writes the finished positions (`--output-embedding`). Relative →
    /// resolved against the config's directory. Ignored by `gui` (its save-embedding
    /// button picks paths interactively).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_embedding: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedder: Option<EmbedderConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attrs: Option<AttrsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme: Option<ThemeConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visual: Option<VisualConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sizing: Option<SizingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export: Option<ExportConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legend: Option<LegendConfig>,
    /// Presence enables density (heatmap) render mode. See [`DensityConfig`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density: Option<DensityConfig>,
}

impl FileConfig {
    /// An all-`None` config — the neutral element for merging (leaves every default
    /// untouched). Same as `FileConfig::default()`; named for readability at call sites.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Read and parse a config file, then resolve every relative path against the config
    /// file's *parent directory* (absolute paths pass through). Surfaces the config path on
    /// I/O errors and toml's line/column info on parse errors (toml's `Display` already
    /// carries the location; the anyhow context just names the file).
    pub fn load(path: &Path) -> Result<FileConfig> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read config file {}", path.display()))?;
        let mut config: FileConfig = toml::from_str(&text)
            .with_context(|| format!("parse config file {}", path.display()))?;

        // Resolve relative paths against the directory the config lives in, so a config +
        // its data files form a relocatable bundle. `parent()` is `Some("")` for a bare
        // filename (CWD), which `resolve_rel` treats as a no-op.
        let base = path.parent().unwrap_or_else(|| Path::new(""));
        resolve_rel(&mut config.graph, base);
        resolve_rel(&mut config.input_embedding, base);
        resolve_rel(&mut config.output_embedding, base);
        if let Some(attrs) = config.attrs.as_mut() {
            resolve_rel(&mut attrs.colors, base);
            resolve_rel(&mut attrs.labels, base);
            resolve_rel(&mut attrs.categories, base);
            resolve_rel(&mut attrs.edge_colors, base);
        }
        if let Some(theme) = config.theme.as_mut() {
            resolve_rel(&mut theme.dir, base);
        }
        if let Some(text) = config.text.as_mut() {
            resolve_rel(&mut text.font, base);
        }
        if let Some(legend) = config.legend.as_mut() {
            resolve_rel(&mut legend.names, base);
            resolve_rel(&mut legend.entries_from, base);
        }

        // `[camera]` is only meaningful as a complete (center, half) pair. Reject a
        // half-specified table rather than silently dropping one axis of the viewport.
        if let Some(camera) = config.camera.as_ref() {
            if camera.center.is_some() != camera.half.is_some() {
                bail!(
                    "config file {}: [camera] needs both center and half (got only one)",
                    path.display()
                );
            }
        }

        Ok(config)
    }

    /// Serialize to a pretty TOML string, prefixed with a one-line header comment.
    /// `skip_serializing_if` drops unset fields, so an [`empty`](Self::empty) config
    /// produces just the header (no section tables).
    pub fn to_toml_string(&self) -> Result<String> {
        let body = toml::to_string_pretty(self).context("serialize config to TOML")?;
        Ok(format!("# wembed-viewer config\n{body}"))
    }

    /// Write the config to disk (see [`to_toml_string`](Self::to_toml_string)), naming the
    /// path on any I/O error.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = self.to_toml_string()?;
        std::fs::write(path, text)
            .with_context(|| format!("write config file {}", path.display()))?;
        Ok(())
    }
}

/// Rewrite a relative path in place to `base/<path>`; leave absolute paths (and `None`)
/// untouched. A blank `base` (bare-filename config) is a no-op — `join` on `""` is
/// identity here because we only rewrite *relative* paths, and joining a relative path
/// onto `""` yields the same relative path.
fn resolve_rel(path: &mut Option<PathBuf>, base: &Path) {
    if let Some(p) = path.as_ref() {
        if p.is_relative() && !base.as_os_str().is_empty() {
            *path = Some(base.join(p));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::{NamedTempFile, TempDir};

    /// Write `contents` to a temp file and return the handle (kept alive by the caller).
    fn temp_with(contents: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("create temp file");
        f.write_all(contents.as_bytes()).expect("write temp file");
        f.flush().expect("flush temp file");
        f
    }

    /// A config exercising every section and field. Kept as a fixture so the full-parse and
    /// round-trip tests share one source of truth.
    const FULL_CONFIG: &str = r##"
graph = "g.edg"
input_embedding = "e.emb"
output_embedding = "out.emb"

[embedder]
iterations = 10000
seed = 42
dimension = 8
dimension_hint = 3.0
learning_rate = 10.0
lr_cooling_factor = 0.99
layered = false
centre = 0.0
dyn_buffer = -1.0
lr_schedule = "loss-adaptive"
warmup_steps = 5
lr_decay_factor = 0.6
lr_decay_threshold = 0.02
lr_adapt_patience = 12
lr_growth_factor = 1.1
lr_growth_threshold = 0.15
stop_criterion = "displacement"
stop_displacement_tol = 0.002
stop_displacement_patience = 7
stop_loss_tol = 0.0005
stop_loss_patience = 40

[attrs]
colors = "c.txt"
labels = "l.jsonl"
categories = "cat.txt"
edge_colors = "ec.txt"

[theme]
dir = "themes/"
name = "tokyo"

[visual]
vertex_color_source = "categories"
vertex_size_source = "weight"
vertex_size_scale = 1.0
vertex_alpha_scale = 1.0
edge_alpha_source = "weight"
edge_alpha_scale = 0.35
edge_width_scale = 0.5
edge_color_source = "length"
edge_length_range = [0.25, 4.0]
projection = "first-two"
depth_fade = 0.5
flip_x = true
flip_y = false
search = "widget"

[sizing]
density_factor = 0.5
point_min_pixels = 3.0
point_max_pixels = 40.0

[camera]
center = [1.0, 2.0]
half = 50.0
rotation_deg = 0.0

[export]
width = 3840
height = 2160

[legend]
size = 22.0
anchor = "bottom-right"
color = "#aabbcc"

[[legend.entries]]
label = "hubs"
category = 0

[[legend.entries]]
label = "manual"
color = "#ff0000"

[density]
sigma = 1.5
curve = "log"
knee_percentile = 50.0
white_percentile = 99.5
ramp = ["#0b0e14", "#552200", "#ff9940", "#fff2e0"]
"##;

    // ---- full / partial / empty parse ----

    #[test]
    fn full_config_parses_and_spot_checks() {
        let f = temp_with(FULL_CONFIG);
        let cfg = FileConfig::load(f.path()).expect("parse ok");

        // Top-level paths were relative → resolved against the temp file's parent dir.
        let base = f.path().parent().unwrap();
        assert_eq!(cfg.graph.as_deref(), Some(base.join("g.edg").as_path()));
        assert_eq!(
            cfg.input_embedding.as_deref(),
            Some(base.join("e.emb").as_path())
        );
        assert_eq!(
            cfg.output_embedding.as_deref(),
            Some(base.join("out.emb").as_path())
        );

        let emb = cfg.embedder.as_ref().expect("embedder section");
        assert_eq!(emb.iterations, Some(10_000));
        assert_eq!(emb.seed, Some(42));
        assert_eq!(emb.dimension, Some(8));
        assert_eq!(emb.dimension_hint, Some(3.0));
        assert_eq!(emb.learning_rate, Some(10.0));
        assert_eq!(emb.lr_cooling_factor, Some(0.99));
        assert_eq!(emb.layered, Some(false));
        assert_eq!(emb.centre, Some(0.0));
        assert_eq!(emb.dyn_buffer, Some(-1.0));
        assert_eq!(emb.lr_schedule, Some(LrSchedule::LossAdaptive));
        assert_eq!(emb.warmup_steps, Some(5));
        assert_eq!(emb.lr_decay_factor, Some(0.6));
        assert_eq!(emb.lr_decay_threshold, Some(0.02));
        assert_eq!(emb.lr_adapt_patience, Some(12));
        assert_eq!(emb.lr_growth_factor, Some(1.1));
        assert_eq!(emb.lr_growth_threshold, Some(0.15));
        assert_eq!(emb.stop_criterion, Some(StopCriterion::Displacement));
        assert_eq!(emb.stop_displacement_tol, Some(0.002));
        assert_eq!(emb.stop_displacement_patience, Some(7));
        assert_eq!(emb.stop_loss_tol, Some(0.0005));
        assert_eq!(emb.stop_loss_patience, Some(40));

        let attrs = cfg.attrs.as_ref().expect("attrs section");
        assert_eq!(attrs.colors.as_deref(), Some(base.join("c.txt").as_path()));
        assert_eq!(
            attrs.edge_colors.as_deref(),
            Some(base.join("ec.txt").as_path())
        );

        let theme = cfg.theme.as_ref().expect("theme section");
        assert_eq!(theme.name.as_deref(), Some("tokyo"));

        let vis = cfg.visual.as_ref().expect("visual section");
        assert_eq!(vis.vertex_color_source, Some(VertexColorSource::Categories));
        assert_eq!(vis.edge_alpha_scale, Some(0.35));
        assert_eq!(vis.edge_color_source, Some(EdgeColorSource::Length));
        assert_eq!(vis.edge_length_range, Some([0.25, 4.0]));
        assert_eq!(vis.projection, Some(ProjectionMode::FirstTwo));
        assert_eq!(vis.depth_fade, Some(0.5));
        assert_eq!(vis.flip_x, Some(true));
        assert_eq!(vis.flip_y, Some(false));
        assert_eq!(vis.search.as_deref(), Some("widget"));

        let sizing = cfg.sizing.as_ref().expect("sizing section");
        assert_eq!(sizing.density_factor, Some(0.5));
        assert_eq!(sizing.point_max_pixels, Some(40.0));

        let cam = cfg.camera.as_ref().expect("camera section");
        assert_eq!(cam.center, Some([1.0, 2.0]));
        assert_eq!(cam.half, Some(50.0));
        assert_eq!(cam.rotation_deg, Some(0.0));

        let export = cfg.export.as_ref().expect("export section");
        assert_eq!(export.width, Some(3840));
        assert_eq!(export.height, Some(2160));

        let legend = cfg.legend.as_ref().expect("legend section");
        assert_eq!(legend.size, Some(22.0));
        assert_eq!(legend.anchor, Some(crate::text::Anchor::BottomRight));
        assert_eq!(legend.color.as_deref(), Some("#aabbcc"));
        assert_eq!(legend.entries.len(), 2);
        assert_eq!(legend.entries[0].label, "hubs");
        assert_eq!(legend.entries[0].category, Some(0));
        assert!(legend.entries[0].color.is_none());
        assert_eq!(legend.entries[1].color.as_deref(), Some("#ff0000"));
        assert!(legend.entries[1].category.is_none());

        let density = cfg.density.as_ref().expect("density section");
        assert_eq!(density.sigma, Some(1.5));
        assert_eq!(density.curve.as_deref(), Some("log"));
        assert_eq!(density.knee_percentile, Some(50.0));
        assert_eq!(density.white_percentile, Some(99.5));
        assert_eq!(
            density.ramp.as_deref(),
            Some(&["#0b0e14", "#552200", "#ff9940", "#fff2e0"].map(String::from)[..])
        );
    }

    #[test]
    fn partial_config_only_visual_two_fields() {
        let f = temp_with("[visual]\nvertex_size_scale = 2.0\nedge_alpha_scale = 0.5\n");
        let cfg = FileConfig::load(f.path()).expect("parse ok");

        assert!(cfg.graph.is_none());
        assert!(cfg.input_embedding.is_none());
        assert!(cfg.embedder.is_none());
        assert!(cfg.attrs.is_none());
        assert!(cfg.theme.is_none());
        assert!(cfg.sizing.is_none());
        assert!(cfg.camera.is_none());
        assert!(cfg.export.is_none());

        let vis = cfg.visual.as_ref().expect("visual section present");
        assert_eq!(vis.vertex_size_scale, Some(2.0));
        assert_eq!(vis.edge_alpha_scale, Some(0.5));
        // Untouched visual fields stay None.
        assert!(vis.vertex_color_source.is_none());
        assert!(vis.vertex_size_source.is_none());
        assert!(vis.search.is_none());
    }

    // ---- deny_unknown_fields ----

    #[test]
    fn unknown_top_level_key_errors() {
        let f = temp_with("nonsense = 1\n");
        let err = FileConfig::load(f.path()).unwrap_err().to_string();
        // anyhow context names the file; the source (printed by the caller) carries the key.
        assert!(err.contains("parse config file"), "unexpected error: {err}");
    }

    #[test]
    fn unknown_key_inside_section_errors() {
        let f = temp_with("[visual]\nbogus = true\n");
        assert!(FileConfig::load(f.path()).is_err());
    }

    // ---- path resolution ----

    #[test]
    fn relative_paths_resolved_against_config_dir() {
        // Use a real dir so `parent()` is a concrete directory, not the temp root.
        let dir = TempDir::new().expect("temp dir");
        let cfg_path = dir.path().join("session.toml");
        std::fs::write(&cfg_path, "graph = \"sub/g.edg\"\n").expect("write config");

        let cfg = FileConfig::load(&cfg_path).expect("parse ok");
        assert_eq!(
            cfg.graph.as_deref(),
            Some(dir.path().join("sub/g.edg").as_path()),
            "relative graph path should be joined onto the config's dir"
        );
    }

    #[test]
    fn absolute_paths_pass_through_untouched() {
        let dir = TempDir::new().expect("temp dir");
        let cfg_path = dir.path().join("session.toml");
        // An absolute path that is clearly not under `dir`.
        std::fs::write(&cfg_path, "graph = \"/etc/graph.edg\"\n").expect("write config");

        let cfg = FileConfig::load(&cfg_path).expect("parse ok");
        assert_eq!(cfg.graph.as_deref(), Some(Path::new("/etc/graph.edg")));
    }

    // ---- [camera] pairing ----

    #[test]
    fn camera_half_specified_pair_errors_either_direction() {
        // Both orderings hit the same `center.is_some() != half.is_some()` guard.
        for body in ["[camera]\ncenter = [0.0, 0.0]\n", "[camera]\nhalf = 50.0\n"] {
            let f = temp_with(body);
            let err = FileConfig::load(f.path()).unwrap_err().to_string();
            assert!(
                err.contains("[camera] needs both center and half"),
                "unexpected error: {err}"
            );
        }
    }

    #[test]
    fn camera_both_present_ok() {
        let f = temp_with("[camera]\ncenter = [0.0, 0.0]\nhalf = 50.0\n");
        let cfg = FileConfig::load(f.path()).expect("both present parses");
        let cam = cfg.camera.expect("camera section");
        assert_eq!(cam.center, Some([0.0, 0.0]));
        assert_eq!(cam.half, Some(50.0));
        assert!(cam.rotation_deg.is_none());
    }

    #[test]
    fn camera_rotation_only_ok() {
        // rotation_deg is independent of the center/half pair — a rotation-only [camera]
        // is valid (consumer auto-fits center/half, applies the rotation on .emb replay).
        let f = temp_with("[camera]\nrotation_deg = 30.0\n");
        let cfg = FileConfig::load(f.path()).expect("rotation-only parses");
        let cam = cfg.camera.expect("camera section");
        assert!(cam.center.is_none());
        assert!(cam.half.is_none());
        assert_eq!(cam.rotation_deg, Some(30.0));
    }

    // ---- legend entries_from ----

    #[test]
    fn legend_entries_from_key_resolves_against_config_dir() {
        let dir = TempDir::new().expect("temp dir");
        let cfg_path = dir.path().join("session.toml");
        std::fs::write(&cfg_path, "[legend]\nentries_from = \"legend.toml\"\n").expect("write cfg");

        let cfg = FileConfig::load(&cfg_path).expect("parse ok");
        let legend = cfg.legend.expect("legend section");
        assert_eq!(
            legend.entries_from.as_deref(),
            Some(dir.path().join("legend.toml").as_path())
        );
        assert!(legend.names.is_none());
        assert!(legend.entries.is_empty());
    }

    #[test]
    fn external_entries_file_loads_and_validates() {
        let f = temp_with(
            "[[entries]]\nlabel = \"hubs\"\ncategory = 0\n\n[[entries]]\nlabel = \"manual\"\ncolor = \"#ff0000\"\n",
        );
        let entries = LegendEntriesFile::load(f.path()).expect("load ok");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].label, "hubs");
        assert_eq!(entries[0].category, Some(0));
        assert_eq!(entries[1].color.as_deref(), Some("#ff0000"));
    }

    #[test]
    fn external_entries_file_unknown_key_errors() {
        let f = temp_with("[[entries]]\nlabel = \"x\"\ncategory = 0\nbogus = true\n");
        assert!(LegendEntriesFile::load(f.path()).is_err());
    }

    #[test]
    fn external_entries_file_unknown_top_level_key_errors() {
        let f = temp_with("stray = 1\n[[entries]]\nlabel = \"x\"\ncategory = 0\n");
        assert!(LegendEntriesFile::load(f.path()).is_err());
    }

    // ---- serialize ----

    #[test]
    fn full_config_round_trips_through_toml() {
        // Serialize-then-reparse must reproduce the config exactly. Also guards the toml
        // crate's value-after-table rule: a scalar field ordered after `Vec<...>` in any
        // section struct makes serialization fail outright.
        let f = temp_with(FULL_CONFIG);
        let cfg = FileConfig::load(f.path()).expect("parse ok");
        let text = cfg.to_toml_string().expect("serialize ok");
        let reparsed: FileConfig = toml::from_str(&text).expect("reparse ok");
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn empty_config_serializes_without_section_headers() {
        let text = FileConfig::empty()
            .to_toml_string()
            .expect("serialize empty");
        // Only the header comment; skip_serializing_if drops every field/table.
        assert!(!text.contains('['), "no section headers expected: {text:?}");
        assert!(!text.contains('='), "no key/value lines expected: {text:?}");
        assert!(text.contains("# wembed-viewer config"));
    }
}
