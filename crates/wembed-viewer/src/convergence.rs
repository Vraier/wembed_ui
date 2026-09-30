//! Convergence panel: a floating "Convergence" window that makes the embedder's stopping
//! behavior legible live. The layout is a grid of **small multiples** — one COLUMN per
//! layered-embedder layer, four metric ROWS per column (loss, learning rate, displacement,
//! loss rate) — so each layer's convergence shape can be compared side by side, each starting
//! from its own iteration 0.
//!
//! Owns the drained [`MetricSample`] history and the resolved embedder knobs (fixed for the
//! session — the panel is read-only, nothing here is interactive). Samples come from the live embedder worker (see [`crate::embedder`]);
//! `.emb` replay produces none, so the window doesn't appear there. A non-layered run is
//! simply the N=1 case: one column spanning the panel.
//!
//! Design: real values are readable, not normalized. Rows A (loss), B (learning rate) and C
//! (displacement) are drawn in log10 with a `1e{n}` axis formatter; non-positive samples (a
//! converged layer reporting exactly 0, whose log is undefined) are dropped, leaving a gap
//! rather than a fabricated floor spike. Row D is the loss `rate(t)` (windowed relative loss
//! decrease, the loss stop's ftol signal) on a SYMLOG axis in units of ftol — linear through
//! 0, log beyond, sign kept — so its threshold crossing is legible and the negative
//! post-expansion dips don't blow up to −∞. Every cell gets its OWN robust y-range, recomputed
//! live from a [`Q_LO`, `Q_HI`] quantile band of its points (see [`cell_bounds`]) so the big
//! transient spikes at each layer's first steps don't flatten the readable range; the trade is
//! that magnitudes aren't directly comparable across columns (shapes still are). Each stop
//! tolerance is drawn as a dashed `HLine` straight from settings, folded into the bounds so it
//! stays on screen — watching a signal cross its line IS the criterion firing.

use egui::Color32;
use egui_plot::{HLine, Line, LineStyle, Plot};

use crate::cli::{LrSchedule, StopCriterion};
use crate::embedder::{layer_label, MetricSample};
use crate::settings::EmbedderParams;

/// Target point count per drawn series (per layer) — long runs are stride-decimated down
/// to this so per-frame tessellation stays bounded. Full history is kept for export.
const DRAW_POINTS: usize = 2000;
/// Minimum width of a layer column; past this the columns horizontally scroll rather than
/// shrink into unreadable slivers. Each column now carries its own y-axis, so a touch wider.
const MIN_COL_W: f32 = 210.0;
/// Fixed width reserved for every cell's y-axis so the plot *areas* line up across a column
/// even when tick labels differ in length (`1e-3` vs `1e-12` vs a symlog `3e-4`). Without it,
/// egui sizes each y-axis to its own labels and the stacked plots misalign.
const Y_AXIS_W: f32 = 44.0;
/// Lower/upper quantile fractions bounding each cell's y-axis, plus the padding fraction of
/// the resulting span. The band ignores the transient spikes at a layer's first steps (the
/// coarsening outliers) that would otherwise flatten the readable range.
const Q_LO: f64 = 0.02;
const Q_HI: f64 = 0.98;
const Y_MARGIN: f64 = 0.1;

// Curve colors. Kept distinct so the three rows read at a glance; each tolerance line
// shares its signal's hue (dashed). A single color key at the top stands in for per-cell
// legends, which would be unreadable clutter across a grid.
const C_LOSS: Color32 = Color32::from_rgb(0xff, 0x8a, 0x65);
const C_DISP: Color32 = Color32::from_rgb(0x4f, 0xc3, 0xf7);
const C_LR: Color32 = Color32::from_rgb(0xb3, 0x9d, 0xdb);
const C_IMPR: Color32 = Color32::from_rgb(0x9c, 0xcc, 0x65);
const C_LAYER: Color32 = Color32::from_rgb(0xbd, 0xbd, 0xbd);

/// A contiguous run of samples that ran on one layer. Borrows into the sample history;
/// layers never interleave, so a slice is exact.
struct LayerSegment<'a> {
    /// 0-based order of appearance, for egui ids.
    idx: usize,
    /// wembed's layer (counts down to 0) and layer count, for the label.
    layer: i32,
    num_layers: i32,
    /// Vertex count of the layer.
    n: u32,
    /// Global iteration of the segment's first sample — subtracted to reset x to 0.
    start_iter: u64,
    samples: &'a [MetricSample],
}

pub struct Convergence {
    /// Per-step samples for the whole run. Grows unbounded — at the 10k-iteration default
    /// that's a few hundred KB, so no ring buffer; drawing decimates instead.
    samples: Vec<MetricSample>,
    params: EmbedderParams,
}

impl Convergence {
    pub fn new(params: EmbedderParams) -> Self {
        Self {
            samples: Vec::new(),
            params,
        }
    }

    /// Drain target for `TopologyProducer::drain_metrics` — called once per tick.
    pub fn samples_mut(&mut self) -> &mut Vec<MetricSample> {
        &mut self.samples
    }

    /// Draw the floating convergence window. Renders nothing while no samples exist (graph
    /// still loading, or an `.emb` replay that never produces any) — callers don't gate.
    ///
    /// Pinned top-right via `anchor` and collapsed on first show (`default_open(false)`):
    /// the title bar's own collapse triangle opens/closes it in place, so it starts out of
    /// the way and the user expands it on demand. Resizable by edge-drag once open — layered
    /// runs need a large panel to stay readable.
    pub fn window(&self, ctx: &egui::Context) {
        if self.samples.is_empty() {
            return;
        }
        egui::Window::new("Convergence")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
            .default_open(false)
            .default_size(egui::vec2(560.0, 460.0))
            .min_size(egui::vec2(340.0, 280.0))
            .resizable(true)
            .show(ctx, |ui| self.contents(ui));
    }

    fn contents(&self, ui: &mut egui::Ui) {
        let last = self
            .samples
            .last()
            .copied()
            .expect("non-empty checked by caller");
        self.readout(ui, &last);
        ui.separator();

        let segs = self.layer_segments();
        if segs.is_empty() {
            return;
        }

        // Row heights: all four cells equal, derived from the window's available height so
        // the whole grid fits without the window fighting its own content.
        let spacing = ui.spacing().item_spacing.y;
        let header_h = ui.text_style_height(&egui::TextStyle::Small) + spacing;
        let plots_h = (ui.available_height() - header_h).max(150.0);
        let cell_h = ((plots_h - 3.0 * spacing) / 4.0).max(58.0);

        // Column width: fill when a few layers, floor at MIN_COL_W (then scroll) when many.
        let n = segs.len() as f32;
        let spacing_x = ui.spacing().item_spacing.x;
        let fit_w = ((ui.available_width() - (n - 1.0) * spacing_x) / n).floor();
        let col_w = fit_w.max(MIN_COL_W);

        egui::ScrollArea::horizontal().show(ui, |ui| {
            ui.horizontal_top(|ui| {
                for seg in &segs {
                    ui.vertical(|ui| {
                        ui.set_width(col_w);
                        let label = if seg.num_layers > 1 {
                            format!(
                                "{} · {}",
                                layer_label(seg.layer, seg.num_layers),
                                fmt_count(seg.n)
                            )
                        } else {
                            format!("{} vertices", fmt_count(seg.n))
                        };
                        ui.label(egui::RichText::new(label).small().color(C_LAYER))
                            .on_hover_text(format!("{} vertices", seg.n));
                        self.draw_column(ui, seg, col_w, cell_h);
                    });
                }
            });
        });
    }

    /// One layer's column: four equal-height metric cells, x reset to per-layer iteration.
    /// The cells share a per-column link group so a hover cursor aligns across metrics;
    /// columns stay independent on x (different per-layer ranges). Each cell has its own
    /// quantile-derived y-scale; x-axis labels only on the bottom (D) row.
    ///
    /// Rows: A = loss (the objective) · B = learning rate (the schedule driving it) — both
    /// context, no tolerance line; C = displacement + its tolerance (displacement criterion);
    /// D = loss rate + its ftol (loss criterion), on a symlog axis (see [`symlog`]).
    fn draw_column(&self, ui: &mut egui::Ui, seg: &LayerSegment, w: f32, cell_h: f32) {
        let e = &self.params;
        let link = egui::Id::new(("convergence_col", seg.idx));

        // log10 of each signal; non-positive samples (a converged layer reporting exactly 0,
        // whose log is undefined) are dropped by `seg_series` rather than floored to a fake
        // spike — a gap reads as "off the bottom", which is the truth.
        let loss = self.seg_series(seg, |m| m.loss.log10());
        let lr = self.seg_series(seg, |m| m.learning_rate.log10());
        let disp = self.seg_series(seg, |m| m.rel_displacement.log10());
        // rate(t) on a symlog axis in ftol units: linear through 0, log beyond, sign kept —
        // so (the +inf warmup dropped by seg_series aside) the negative post-expansion dips
        // and the tiny sub-ftol tail are both visible without a −∞ blowup.
        let l = symlog_linthresh(e.stop_loss_tol);
        let rate = self.seg_series(seg, |m| symlog(m.rel_loss_improvement, l));

        let disp_tol = e.stop_displacement_tol.log10();

        // A — loss (log10). The objective; no tolerance line.
        log_cell(
            ("conv_loss", seg.idx),
            link,
            w,
            cell_h,
            cell_bounds(&[&loss], &[]),
        )
        .show_axes([false, true])
        .show(ui, |pui| {
            pui.line(Line::new("loss", loss).color(C_LOSS).width(2.0));
        });

        // B — learning rate (log10). The schedule driving the descent; no tolerance line.
        log_cell(
            ("conv_lr", seg.idx),
            link,
            w,
            cell_h,
            cell_bounds(&[&lr], &[]),
        )
        .show_axes([false, true])
        .show(ui, |pui| {
            pui.line(Line::new("learning rate", lr).color(C_LR).width(1.5));
        });

        // C — displacement (bold), log10. HLine at the displacement tolerance (folded into
        // the bounds so it stays visible): watching `disp` descend to it IS the displacement
        // criterion firing.
        log_cell(
            ("conv_disp", seg.idx),
            link,
            w,
            cell_h,
            cell_bounds(&[&disp], &[disp_tol]),
        )
        .show_axes([false, true])
        .show(ui, |pui| {
            pui.line(Line::new("displacement", disp).color(C_DISP).width(2.0));
            pui.hline(
                HLine::new("displacement tol", disp_tol)
                    .color(C_DISP)
                    .width(1.0)
                    .style(LineStyle::dashed_loose()),
            );
        });

        // D — loss rate (symlog in ftol units). Descends toward the ftol line as the loss
        // flattens; dips below 0 right after a layer expansion (loss briefly rises). Once it
        // stays inside ±ftol for stop_loss_patience steps the loss criterion fires. Bounds
        // fold in both tol lines so the band stays on screen.
        let tol_t = symlog(e.stop_loss_tol, l);
        let (lo, hi) = cell_bounds(&[&rate], &[tol_t, -tol_t]);
        Plot::new(("conv_rate", seg.idx))
            .link_axis(link, [true, false])
            .link_cursor(link, [true, false])
            .width(w)
            .height(cell_h)
            .y_axis_min_width(Y_AXIS_W)
            .show_axes([true, true])
            .default_y_bounds(lo, hi)
            .y_axis_formatter(move |mark, _| fmt_symlog_tick(mark.value, l))
            .show(ui, |pui| {
                pui.line(Line::new("loss rate", rate).color(C_IMPR).width(1.5));
                for t in [tol_t, -tol_t] {
                    pui.hline(
                        HLine::new("loss rate tol", t)
                            .color(C_IMPR)
                            .width(1.0)
                            .style(LineStyle::dashed_loose()),
                    );
                }
            });
    }

    /// One row per plot row, same names and order; the names carry the curve colours, so
    /// the rows double as the legend. The below-tol counts are derived here, not read from
    /// C++: they're definitional (consecutive trailing steps meeting the condition) and
    /// exactly reproduce the C++ monitors because `rel_displacement` and
    /// `rel_loss_improvement` (= `rate(t)`) already fold in the EMA + window the monitors use.
    fn readout(&self, ui: &mut egui::Ui, last: &MetricSample) {
        use egui::RichText;
        let e = &self.params;
        ui.horizontal_wrapped(|ui| {
            ui.label("stop criterion:");
            ui.strong(criterion_label(e.stop_criterion));
            ui.label(RichText::new("· dashed = tolerance").small().weak());
        });
        let row = |ui: &mut egui::Ui, name: &str, color: Color32, value: String, rest: String| {
            ui.horizontal_wrapped(|ui| {
                let name = ui.label(RichText::new(name).color(color));
                ui.strong(value);
                ui.label(rest);
                name
            })
            .inner
        };

        row(
            ui,
            "loss",
            C_LOSS,
            format!("{:.2e}", last.loss),
            String::new(),
        );
        row(
            ui,
            "learning rate",
            C_LR,
            format!("{:.2e}", last.learning_rate),
            schedule_summary(e),
        );
        let disp_k = self.trailing_streak(|m| m.rel_displacement < e.stop_displacement_tol);
        row(
            ui,
            "displacement",
            C_DISP,
            format!("{:.2e}", last.rel_displacement),
            format!(
                "· tol {:.1e} · below tol {disp_k}/{}",
                e.stop_displacement_tol, e.stop_displacement_patience
            ),
        );
        // two-sided like the C++ monitor (a rising loss is not stagnation); the +inf warmup
        // rate is never below tol, so the streak stays 0 as in C++
        let loss_k = self.trailing_streak(|m| m.rel_loss_improvement.abs() < e.stop_loss_tol);
        let rate = if last.rel_loss_improvement.is_finite() {
            format!("{:.2e}", last.rel_loss_improvement)
        } else {
            "warmup".to_string()
        };
        row(
            ui,
            "loss rate",
            C_IMPR,
            rate,
            format!(
                "· tol {:.1e} · below tol {loss_k}/{}",
                e.stop_loss_tol, e.stop_loss_patience
            ),
        )
        .on_hover_text(
            "Relative loss decrease over a 30-step window. Plotted on a symlog axis (linear \
             near 0, log beyond).",
        );
    }

    /// Number of trailing samples (from the newest backward) that continuously satisfy
    /// `pred`. A single failing step resets the streak — exactly the C++ patience counter.
    fn trailing_streak(&self, pred: impl Fn(&MetricSample) -> bool) -> usize {
        self.samples.iter().rev().take_while(|m| pred(m)).count()
    }

    /// Partition the sample history into contiguous per-layer segments. Layers never
    /// interleave, so each segment is a slice. O(n) per frame; there are only a handful of
    /// segments.
    fn layer_segments(&self) -> Vec<LayerSegment<'_>> {
        let s = &self.samples;
        let mut segs = Vec::new();
        let mut start = 0;
        for i in 1..=s.len() {
            if i == s.len() || s[i].layer != s[start].layer {
                segs.push(LayerSegment {
                    idx: segs.len(),
                    layer: s[start].layer,
                    num_layers: s[start].num_layers,
                    n: s[start].num_vertices,
                    start_iter: s[start].iteration,
                    samples: &s[start..i],
                });
                start = i;
            }
        }
        segs
    }

    /// Stride-decimate one segment to ~`DRAW_POINTS` points, mapping each sample to
    /// `[iteration_within_layer, value]`. The final sample is always kept so the live tip
    /// is exact.
    fn seg_series(
        &self,
        seg: &LayerSegment,
        value: impl Fn(&MetricSample) -> f64,
    ) -> Vec<[f64; 2]> {
        let point = |m: &MetricSample| [(m.iteration - seg.start_iter) as f64, value(m)];
        let stride = (seg.samples.len() / DRAW_POINTS).max(1);
        // Drop non-finite y (log of a non-positive sample, or the +inf warmup rate): it can't
        // be plotted. The line connects across the gap rather than diving to a fake floor.
        let mut out: Vec<[f64; 2]> = seg
            .samples
            .iter()
            .step_by(stride)
            .map(point)
            .filter(|p| p[1].is_finite())
            .collect();
        if stride > 1 {
            if let Some(last) = seg.samples.last() {
                let p = point(last);
                if p[1].is_finite() && out.last() != Some(&p) {
                    out.push(p);
                }
            }
        }
        out
    }
}

/// A log10 metric cell: robust y-bounds (in log units, from [`cell_bounds`]) re-applied
/// each frame so auto-fit can't thrash on the layer jumps, x auto-fit + linked to the
/// column's cursor group, `1e{n}` y labels. The caller sets `show_axes`.
fn log_cell(
    id: (&'static str, usize),
    link: egui::Id,
    w: f32,
    h: f32,
    bounds: (f64, f64),
) -> Plot<'static> {
    Plot::new(id)
        .link_axis(link, [true, false])
        .link_cursor(link, [true, false])
        .width(w)
        .height(h)
        .y_axis_min_width(Y_AXIS_W)
        .default_y_bounds(bounds.0, bounds.1)
        .y_axis_formatter(|mark, _| format!("1e{:.0}", mark.value))
}

/// Nearest-rank quantile of an ascending-sorted, non-empty slice. `q` clamped to `[0, 1]`.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len() - 1;
    let idx = (q.clamp(0.0, 1.0) * last as f64).round() as usize;
    sorted[idx.min(last)]
}

/// Robust y-bounds for one cell: the [`Q_LO`, `Q_HI`] quantile band of every series' finite
/// y-values (so the transient coarsening spikes don't stretch the range), extended to
/// include any `refs` (tolerance lines / baselines that must stay on screen), then padded
/// by [`Y_MARGIN`] of the span. Recomputed each frame; falls back to a unit band when
/// there's nothing finite to bound.
fn cell_bounds(series: &[&[[f64; 2]]], refs: &[f64]) -> (f64, f64) {
    let mut ys: Vec<f64> = series
        .iter()
        .flat_map(|s| s.iter())
        .map(|p| p[1])
        .filter(|v| v.is_finite())
        .collect();
    let (mut lo, mut hi) = if ys.is_empty() {
        (0.0, 1.0)
    } else {
        ys.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        (quantile_sorted(&ys, Q_LO), quantile_sorted(&ys, Q_HI))
    };
    for &r in refs.iter().filter(|r| r.is_finite()) {
        lo = lo.min(r);
        hi = hi.max(r);
    }
    let pad = ((hi - lo).abs() * Y_MARGIN).max(1e-6);
    lo -= pad;
    hi += pad;
    if lo < hi {
        (lo, hi)
    } else {
        let c = 0.5 * (lo + hi);
        (c - 0.5, c + 0.5)
    }
}

/// The active learning-rate schedule and the knobs it reads.
fn schedule_summary(e: &EmbedderParams) -> String {
    let mut out = match e.lr_schedule {
        LrSchedule::Exponential => format!("· cooling ×{} per step", e.lr_cooling_factor),
        LrSchedule::LossAdaptive => format!(
            "· loss-adaptive: ×{} below {:.1e}, ×{} above {:.1e}, patience {}",
            e.lr_decay_factor,
            e.lr_decay_threshold,
            e.lr_growth_factor,
            e.lr_growth_threshold,
            e.lr_adapt_patience
        ),
    };
    if e.warmup_steps > 0 {
        out += &format!(" · warmup {} steps", e.warmup_steps);
    }
    out
}

/// Symmetric-log transform in units of `linthresh`: linear within `±linthresh`, log10 beyond,
/// sign preserved. `symlog(0) = 0`; `symlog(±inf) = ±inf` (dropped downstream). Lets a signed
/// signal that must be read against a tiny positive threshold (loss rate vs ftol) show its
/// threshold crossing without a plain-log −∞ blowup on the negative dips.
fn symlog(y: f64, linthresh: f64) -> f64 {
    y.signum() * (1.0 + y.abs() / linthresh).log10()
}

/// Invert [`symlog`] for a y-axis tick position back to the real value, formatted compactly.
fn fmt_symlog_tick(t: f64, linthresh: f64) -> String {
    let r = t.signum() * linthresh * (10f64.powf(t.abs()) - 1.0);
    if r.abs() < linthresh * 1e-3 {
        "0".to_string()
    } else {
        format!("{r:.0e}")
    }
}

/// The symlog linear threshold: the ftol from settings, or a small positive fallback if it's
/// zero/negative, so the transform stays finite.
fn symlog_linthresh(loss_tol: f64) -> f64 {
    if loss_tol > 0.0 {
        loss_tol
    } else {
        1e-6
    }
}

fn criterion_label(c: StopCriterion) -> &'static str {
    match c {
        StopCriterion::Displacement => "displacement",
        StopCriterion::Loss => "loss",
    }
}

/// Compact vertex-count label for a layer column (`3.4k`, `1.2M`, or the raw number).
fn fmt_count(n: u32) -> String {
    match n {
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.1}k", n as f64 / 1e3),
        n => n.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(iteration: u64, rel_displacement: f64, rel_loss_improvement: f64) -> MetricSample {
        MetricSample {
            iteration,
            learning_rate: 1.0,
            loss: 1.0,
            rel_displacement,
            rel_loss_improvement,
            num_vertices: 10,
            layer: 0,
            num_layers: 1,
        }
    }

    fn conv(samples: Vec<MetricSample>) -> Convergence {
        use clap::Parser;
        let cli = crate::cli::Cli::parse_from(["wembed-viewer", "gui", "unused.edg"]);
        let settings =
            crate::settings::Settings::resolve(cli, crate::config::FileConfig::empty()).unwrap();
        let mut c = Convergence::new(settings.embedder);
        *c.samples_mut() = samples;
        c
    }

    #[test]
    fn trailing_streak_counts_only_the_current_run_and_resets_on_a_break() {
        // The patience readouts drive whether an early stop reads as "criterion fired" vs
        // "capped out", so the counter must reset on a single failing step — not tally all
        // matches ever. Strict `<` for both, matching the C++ ftol test (|rate(t)| < relTol).
        let c = conv(vec![
            sample(1, 5e-3, 0.5),  // disp above tol, fast loss decrease — both fail
            sample(2, 5e-4, 0.0),  // disp settled, rate(t) < loss_tol → both hold …
            sample(3, 5e-4, 0.01), // … still holding
            sample(4, 5e-4, 0.01), // 3 trailing for both
        ]);
        assert_eq!(c.trailing_streak(|m| m.rel_displacement < 1e-3), 3);
        assert_eq!(c.trailing_streak(|m| m.rel_loss_improvement < 3e-2), 3);

        // A break at the newest sample zeroes the streak even with matches behind it.
        let c = conv(vec![sample(1, 5e-4, 0.0), sample(2, 5e-2, 0.9)]);
        assert_eq!(c.trailing_streak(|m| m.rel_displacement < 1e-3), 0);

        // rate(t) = +inf during warmup (STILL_IMPROVING) is never sub-tolerance, so it
        // breaks the loss streak even though the loss is still "improving".
        let c = conv(vec![sample(1, 5e-4, 0.0), sample(2, 5e-4, f64::INFINITY)]);
        assert_eq!(c.trailing_streak(|m| m.rel_loss_improvement < 3e-2), 0);
    }

    #[test]
    fn layer_segments_split_on_layer_and_reset_x_per_layer() {
        // Two layers (wembed counts down): iters 1-2 on layer 1 (N=10), iters 3-4 on layer 0
        // (N=40). Segments must partition the whole run, in order, and carry the start
        // iteration for the per-layer x reset (seg 1's first point is iteration 3 → x 0).
        let on = |layer, n, it| MetricSample {
            layer,
            num_layers: 2,
            num_vertices: n,
            ..sample(it, 0.1, 0.0)
        };
        let c = conv(vec![on(1, 10, 1), on(1, 10, 2), on(0, 40, 3), on(0, 40, 4)]);
        let segs = c.layer_segments();
        assert_eq!(segs.len(), 2);
        assert_eq!(
            (segs[0].layer, segs[0].n, segs[0].samples.len()),
            (1, 10, 2)
        );
        assert_eq!((segs[1].layer, segs[1].n, segs[1].start_iter), (0, 40, 3));
        // x is reset relative to the segment start.
        let xs = c.seg_series(&segs[1], |_| 0.0);
        assert_eq!(xs.first().map(|p| p[0]), Some(0.0));
        assert_eq!(xs.last().map(|p| p[0]), Some(1.0));
    }

    #[test]
    fn cell_bounds_clips_outliers_and_folds_in_refs() {
        // The whole point: a handful of huge transient spikes (coarsening start) must not
        // stretch the readable range — the Q_HI quantile keeps the band near the bulk.
        // Outliers are a sub-Q_HI fraction (3 of 203 ≈ 1.5% < 2%), as they are in practice
        // (a layer's first few steps out of thousands).
        let mut pts: Vec<[f64; 2]> = (0..200).map(|i| [i as f64, i as f64 / 200.0]).collect();
        for k in 0..3 {
            pts.push([200.0 + k as f64, 1000.0]); // outliers well past the bulk
        }
        let (lo, hi) = cell_bounds(&[&pts], &[]);
        assert!(lo < hi);
        assert!(
            hi < 5.0,
            "outliers must be clipped out of the range, got hi={hi}"
        );

        // A reference line below the data (e.g. a tolerance the signal hasn't reached yet)
        // is pulled into the range so it stays visible.
        let (lo2, _) = cell_bounds(&[&pts], &[-5.0]);
        assert!(lo2 <= -5.0, "ref line must be folded in, got lo={lo2}");

        // Degenerate all-equal series still yields a valid (lo < hi) band, never a panic
        // from default_y_bounds.
        let flat: Vec<[f64; 2]> = (0..10).map(|i| [i as f64, 3.0]).collect();
        let (flo, fhi) = cell_bounds(&[&flat], &[]);
        assert!(flo < fhi);
    }

    #[test]
    fn symlog_is_zero_centered_odd_and_invertible() {
        let l = 3e-4;
        // Zero-centered, sign-preserving, monotone above the threshold.
        assert_eq!(symlog(0.0, l), 0.0);
        assert!(symlog(1e-2, l) > symlog(l, l) && symlog(l, l) > 0.0);
        // Odd symmetry: a negative dip mirrors the positive rise (no −∞).
        assert_eq!(symlog(-1e-2, l), -symlog(1e-2, l));
        // The +inf warmup maps off-scale (dropped downstream), never to a finite fake value.
        assert!(symlog(f64::INFINITY, l).is_infinite());
        // fmt_symlog_tick inverts a transformed axis position back to ~the real value.
        let r = 5e-3;
        let back: f64 = fmt_symlog_tick(symlog(r, l), l)
            .parse()
            .expect("scientific parse");
        assert!((back - r).abs() / r < 0.05, "round-trip {back} vs {r}");
    }
}
