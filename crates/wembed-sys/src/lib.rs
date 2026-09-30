//! Low-level cxx bindings to the wembed C++ library.
//!
//! The public wembed API is in `namespace wembed` (see `include/wembed.h` in the wembed repo).

#[cxx::bridge(namespace = "wembed_sys")]
pub mod ffi {
    /// Repulsion query backend. Mirrors `wembed::SpatialIndex` — discriminants must match
    /// the C++ enum (the shim converts by `static_cast`).
    #[repr(i32)]
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum SpatialIndex {
        /// Bundled k-d tree: slower, always available.
        IndexKdTree = 0,
        /// sprk tree (wembed's default): 2 to 16 dimensions.
        IndexSprk = 1,
    }

    /// Gradient-descent optimizer. Mirrors `wembed::OptimizerType`. wembed-viewer doesn't
    /// surface this (always Adam); present so the shared `Options` mirror is complete.
    #[repr(i32)]
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum OptimizerType {
        OptimizerSimple = 0,
        OptimizerAdam = 1,
    }

    /// Learning-rate schedule. Mirrors `wembed::LRSchedule` — discriminants must match the
    /// C++ enum (the shim converts by `static_cast`). Each schedule reads a different
    /// subset of the LR knobs on `Options`; see the field docs there.
    #[repr(i32)]
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum LRSchedule {
        /// Per-step multiplicative cooling (reads `lr_cooling_factor`). wembed's default.
        LRExponentialCooling = 0,
        /// Three-zone controller on the loss-decrease `rate(t)`: grows the LR while it
        /// stays above `lr_growth_threshold`, decays below `lr_decay_threshold`, holds in
        /// between — each after `lr_adapt_patience` in-zone steps (reads `lr_decay_factor`/
        /// `lr_growth_factor` too).
        LRLossAdaptive = 1,
    }

    /// Signal that terminates the run (`max_iterations` is always a hard cap on top).
    /// Mirrors `wembed::StopCriterion` — discriminants must match the C++ enum (the shim
    /// converts by `static_cast`). Each variant reads a different subset of the stop knobs
    /// on `Options`; see the field docs there.
    #[repr(i32)]
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum StopCriterion {
        /// (0) Stop once the relative per-step node movement settles (reads
        /// `stop_displacement_tol` / `stop_displacement_patience`).
        StopDisplacement = 0,
        /// (1) Stop once the smoothed loss stagnates (reads `stop_loss_tol` /
        /// `stop_loss_patience`). wembed's default.
        StopLoss = 1,
    }

    /// Undirected edge. Mirrors `wembed::Edge`; shim converts between the two.
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct Edge {
        pub src: i32,
        pub dst: i32,
    }

    /// Loss components from the most recent embedder step. Mirrors `wembed::Loss`;
    /// shim converts between the two.
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Loss {
        pub attractive: f64,
        pub repulsive: f64,
        pub total: f64,
    }

    /// Progress after the most recent embedder step. Mirrors `wembed::Progress`; shim
    /// converts between the two. Estimates are typically within 2x.
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Progress {
        /// Counts down to 0 (the input graph); always 0 without layering.
        pub layer: i32,
        /// 1 without layering.
        pub num_layers: i32,
        /// Of the current layer.
        pub num_vertices: i32,
        /// Steps done in the current layer.
        pub iteration: i32,
        /// Estimated steps of the current layer, -1 if unknown.
        pub expected_iterations: i32,
        /// Time spent inside `calculate_step` in the current layer (pauses don't count).
        pub layer_seconds: f64,
        /// Estimated time until the embedding finishes, -1 if unknown. Only set in layer 0,
        /// after a few steps, until the run passes its estimate.
        pub eta_seconds: f64,
        /// The current layer has stopped; the next step expands to the next layer.
        pub layer_finished: bool,
    }

    /// Embedder configuration. Mirrors `wembed::Options` field-for-field, in the SAME
    /// order as `include/wembed.h` (cxx maps by name, not offset, but keeping the order
    /// aligned makes the 1:1 check trivial). cxx-generated on the C++ side; the shim
    /// converts to `wembed::Options` in `create_embedder`. Keep the field *list* in lockstep
    /// with the C++ struct (add/rename/remove here + in the shim's two field-for-field maps);
    /// default *values* are not hand-copied — `Options::default()` reads them from C++.
    #[derive(Debug, Clone)]
    pub struct Options {
        // ---- Embedding parameters ----
        pub embedding_dimension: i32,
        pub use_unit_weights: bool,
        pub dimension_hint: f64,
        pub layered_embedding: bool,

        // ---- Force parameters ----
        pub index_type: SpatialIndex,
        /// Slack added to repulsion query radii; index rebuilds are skipped while
        /// accumulated node movement fits in it. Negative = auto (`3 / dimension`,
        /// the benchmarked default); `0` = rebuild every iteration.
        pub dynamic_query_buffer: f64,
        pub centre_scale: f64,
        pub expansion_stretch: f64,

        // ---- Gradient descent parameters ----
        pub optimizer_type: OptimizerType,
        pub max_iterations: i32,
        /// Per-step displacement cap, only read when `optimizer_type == OptimizerSimple`.
        pub simple_opt_max_displacement: f64,

        // ---- Learning-rate schedule (each field notes which schedules read it) ----
        pub lr_schedule: LRSchedule,
        /// Initial learning rate (both schedules).
        pub learning_rate: f64,
        /// Linear LR ramp-up over the first steps (both schedules).
        pub warmup_steps: i32,
        /// Per-step multiplicative decay (LRExponentialCooling only).
        pub lr_cooling_factor: f64,
        /// Multiplicative LR drop on a decay event (LRLossAdaptive only).
        pub lr_decay_factor: f64,
        /// Decay when the loss-decrease `rate(t)` stays below this (LRLossAdaptive only).
        pub lr_decay_threshold: f64,
        /// Consecutive in-zone steps before a decay OR growth event (LRLossAdaptive only —
        /// gates both zones).
        pub lr_adapt_patience: i32,
        /// Multiplicative LR growth while the loss keeps decreasing fast (LRLossAdaptive
        /// only; `1.0` disables growth → pure plateau decay).
        pub lr_growth_factor: f64,
        /// Grow when the loss-decrease `rate(t)` stays above this (LRLossAdaptive only).
        pub lr_growth_threshold: f64,

        // ---- Stopping criterion (max_iterations always caps) ----
        /// Which signal terminates the run. Default `StopLoss`.
        pub stop_criterion: StopCriterion,
        /// Relative per-step node movement (mean displacement / radius of gyration) below
        /// which the layout counts as settled (StopDisplacement only).
        pub stop_displacement_tol: f64,
        /// Settled steps in a row before stopping (StopDisplacement only).
        pub stop_displacement_patience: i32,

        // ---- Loss stagnation stop ----
        /// ftol: converged once `|rate(t)|`, the relative loss decrease over a 30-step window,
        /// stays below this (StopLoss only; LRLossAdaptive reads the same signal).
        pub stop_loss_tol: f64,
        /// Sub-tolerance steps in a row before stopping (StopLoss only).
        pub stop_loss_patience: i32,
    }

    unsafe extern "C++" {
        include!("shim.h");

        // ---------- Defaults ----------

        /// `wembed::Options`'s C++ member-initializer defaults, mapped into the shared
        /// `Options` field-for-field. The single source of truth for default *values* —
        /// `Options::default()` delegates here so the header is never hand-copied.
        fn default_options() -> Options;

        // Opaque wembed types. Held via UniquePtr on the Rust side.
        #[namespace = "wembed"]
        type Graph;
        #[namespace = "wembed"]
        type Embedder;

        // ---------- Graph construction ----------

        // Fallible calls return `Result`: cxx turns the thrown `std::exception` into a
        // `cxx::Exception` carrying `what()`. Without it a throw terminates the process.
        fn graph_from_edges(edges: &[Edge]) -> Result<UniquePtr<Graph>>;
        fn graph_from_file(path: &str, comment: &str, delimiter: &str) -> Result<UniquePtr<Graph>>;

        // ---------- Graph accessors (direct wembed methods where the signature fits) ----------

        #[cxx_name = "getNumVertices"]
        fn num_vertices(self: &Graph) -> i32;
        #[cxx_name = "getNumEdges"]
        fn num_edges(self: &Graph) -> i32;

        /// Flat edge list; length == num_edges().
        fn graph_edge_list(g: &Graph) -> UniquePtr<CxxVector<Edge>>;

        // ---------- Embedder construction ----------

        /// Errors on invalid options, e.g. a dimension outside sprk's 2..=16.
        fn create_embedder(g: &Graph, opts: &Options) -> Result<UniquePtr<Embedder>>;

        // ---------- Embedder driver (non-const wembed methods) ----------

        #[cxx_name = "calculateStep"]
        fn calculate_step(self: Pin<&mut Embedder>);
        #[cxx_name = "calculateEmbedding"]
        fn calculate_embedding(self: Pin<&mut Embedder>);

        // ---------- Embedder accessors (const wembed methods) ----------

        #[cxx_name = "isFinished"]
        fn is_finished(self: &Embedder) -> bool;
        #[cxx_name = "getNumVertices"]
        fn num_vertices(self: &Embedder) -> i32;
        #[cxx_name = "getEmbeddingDimension"]
        fn embedding_dimension(self: &Embedder) -> i32;
        /// Learning rate the optimizer used in the most recent step (before the first step:
        /// the initial LR). Cheap read of tracked state — safe to call every step.
        #[cxx_name = "getCurrentLearningRate"]
        fn current_learning_rate(self: &Embedder) -> f64;
        /// Relative node displacement of the most recent step (mean per-node movement /
        /// radius of gyration) — the signal the displacement stop criterion watches
        /// (stop when `< stop_displacement_tol`). Cheap read of tracked state — safe to
        /// call every step.
        #[cxx_name = "getLastRelDisplacement"]
        fn last_rel_displacement(self: &Embedder) -> f64;
        /// Windowed relative loss-decrease rate `rate(t)` of the most recent step — the signal
        /// the loss stop criterion watches (a step is stagnant when `|rate| < stop_loss_tol`).
        /// Cheap read of tracked state — safe to call every step.
        #[cxx_name = "getLastRelLossImprovement"]
        fn last_rel_loss_improvement(self: &Embedder) -> f64;

        /// Loss from the most recent step. Tracked during the force computation, so this
        /// is a cheap read — safe to call every step. (Shim fn: `wembed::Loss` is
        /// returned by value, which cxx can't express on an opaque method.)
        fn embedder_loss(e: &Embedder) -> Loss;

        /// Progress after the most recent step. Cheap — safe to call every step.
        fn embedder_progress(e: &Embedder) -> Progress;

        /// Zero-allocation flat copy into a caller-owned buffer, row-major.
        /// `out.len()` MUST equal `num_vertices() * embedding_dimension()`.
        /// This is the hot-path accessor: safe to call every frame at scale.
        fn embedder_copy_coords(e: &Embedder, out: &mut [f64]);

        fn embedder_weights(e: &Embedder) -> UniquePtr<CxxVector<f64>>;

        /// Fresh Graph snapshot (LayeredEmbedder reallocates per call — cheap to skip until needed).
        fn embedder_current_graph(e: &Embedder) -> UniquePtr<Graph>;

        /// Pretty-printed timing tree (goes through wembed::timingsToString). Punt on the raw
        /// TimingResult vector for now — it contains std::string which is awkward across cxx.
        fn embedder_timings_str(e: &Embedder) -> String;

        // ---------- Free helpers ----------

        fn set_seed(seed: i32);
    }
}

// Re-export the bridge types under stable, snake-case Rust names.
pub use ffi::{
    Edge, LRSchedule, Loss, OptimizerType, Options, Progress, SpatialIndex, StopCriterion,
};
// Re-export cxx's smart pointer so downstream crates can name it without depending on cxx
// directly. `wembed_sys::UniquePtr<ffi::Embedder>` is the type wembed-viewer's worker holds.
pub use cxx::UniquePtr;

impl Default for Options {
    /// The C++ header is the single source of truth for default *values*: this delegates to
    /// the shim's [`ffi::default_options`], which reads a default-constructed `wembed::Options`
    /// field-for-field. No values are hand-copied here — change a default in `include/wembed.h`
    /// and it flows straight through. (The viewer's compile-time `settings::defaults` consts,
    /// which exist only so clap's `[default: N]` help strings have a literal, are guarded
    /// against this by `settings::tests::defaults_track_wembed`.)
    fn default() -> Self {
        ffi::default_options()
    }
}
