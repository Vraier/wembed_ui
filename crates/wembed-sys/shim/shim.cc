// Shim implementation. Includes the cxx-generated header for the shared types.

#include "shim.h"

#include <stdexcept>
#include <string>

#include "wembed-sys/src/lib.rs.h"

namespace wembed_sys {

namespace { 

// The cxx-generated enums and their wembed:: counterparts share underlying integer
// values (SpatialIndex 0/1, OptimizerType 0/1, LRSchedule 0/1, StopCriterion 0/1), so
// each conversion is a static_cast. Named helpers for clarity.
wembed::SpatialIndex to_wembed(SpatialIndex s) {
    return static_cast<wembed::SpatialIndex>(static_cast<int32_t>(s));
}

wembed::OptimizerType to_wembed(OptimizerType o) {
    return static_cast<wembed::OptimizerType>(static_cast<int32_t>(o));
}

wembed::LRSchedule to_wembed(LRSchedule s) {
    return static_cast<wembed::LRSchedule>(static_cast<int32_t>(s));
}

wembed::StopCriterion to_wembed(StopCriterion c) {
    return static_cast<wembed::StopCriterion>(static_cast<int32_t>(c));
}

// Field-for-field, in wembed.h order. Every field of wembed::Options is set here — a
// missing one would silently keep the C++ default and diverge from the Rust mirror.
wembed::Options to_wembed(const Options& o) {
    wembed::Options w;
    w.embeddingDimension       = o.embedding_dimension;
    w.useUnitWeights           = o.use_unit_weights;
    w.dimensionHint            = o.dimension_hint;
    w.layeredEmbedding         = o.layered_embedding;
    w.indexType                = to_wembed(o.index_type);
    w.dynamicQueryBuffer       = o.dynamic_query_buffer;
    w.centreScale              = o.centre_scale;
    w.expansionStretch         = o.expansion_stretch;
    w.optimizerType            = to_wembed(o.optimizer_type);
    w.maxIterations            = o.max_iterations;
    w.simpleOptMaxDisplacement = o.simple_opt_max_displacement;
    w.lrSchedule               = to_wembed(o.lr_schedule);
    w.learningRate             = o.learning_rate;
    w.warmupSteps              = o.warmup_steps;
    w.lrCoolingFactor          = o.lr_cooling_factor;
    w.lrDecayFactor            = o.lr_decay_factor;
    w.lrDecayThreshold         = o.lr_decay_threshold;
    w.lrAdaptPatience          = o.lr_adapt_patience;
    w.lrGrowthFactor           = o.lr_growth_factor;
    w.lrGrowthThreshold        = o.lr_growth_threshold;
    w.stopCriterion            = to_wembed(o.stop_criterion);
    w.stopDisplacementTol      = o.stop_displacement_tol;
    w.stopDisplacementPatience = o.stop_displacement_patience;
    w.stopLossTol              = o.stop_loss_tol;
    w.stopLossPatience         = o.stop_loss_patience;
    return w;
}

}  // namespace

// ---------- Graph construction ----------

std::unique_ptr<wembed::Graph> graph_from_edges(rust::Slice<const Edge> edges) {
    std::vector<wembed::Edge> v;
    v.reserve(edges.size());
    for (const auto& e : edges) {
        v.push_back(wembed::Edge{e.src, e.dst});
    }
    return std::make_unique<wembed::Graph>(wembed::graphFromEdges(v));
}

std::unique_ptr<wembed::Graph> graph_from_file(rust::Str path, rust::Str comment, rust::Str delimiter) {
    return std::make_unique<wembed::Graph>(
        wembed::graphFromEdgeListFile(std::string(path), std::string(comment), std::string(delimiter)));
}

// ---------- Graph accessors ----------

std::unique_ptr<std::vector<Edge>> graph_edge_list(const wembed::Graph& g) {
    auto src = g.getEdgeList();
    auto out = std::make_unique<std::vector<Edge>>();
    out->reserve(src.size());
    for (const auto& e : src) {
        out->push_back(Edge{e.src, e.dst});
    }
    return out;
}

// ---------- Options defaults ----------

// Surfaces wembed::Options's C++ member initializers (include/wembed.h) so the Rust
// `Options::default()` reads them over FFI instead of hand-copying the values — the single
// source of truth for defaults. Reverse of `to_wembed`; the shared enums and their wembed::
// counterparts share integer values, so each enum is a static_cast back. Field-for-field,
// in wembed.h order — a missing field silently keeps cxx's zero-init instead of the C++
// default, so keep this exhaustive.
Options default_options() {
    const wembed::Options w;
    Options o;
    o.embedding_dimension         = w.embeddingDimension;
    o.use_unit_weights            = w.useUnitWeights;
    o.dimension_hint              = w.dimensionHint;
    o.layered_embedding           = w.layeredEmbedding;
    o.index_type                  = static_cast<SpatialIndex>(static_cast<int32_t>(w.indexType));
    o.dynamic_query_buffer        = w.dynamicQueryBuffer;
    o.centre_scale                = w.centreScale;
    o.expansion_stretch           = w.expansionStretch;
    o.optimizer_type              = static_cast<OptimizerType>(static_cast<int32_t>(w.optimizerType));
    o.max_iterations              = w.maxIterations;
    o.simple_opt_max_displacement = w.simpleOptMaxDisplacement;
    o.lr_schedule                 = static_cast<LRSchedule>(static_cast<int32_t>(w.lrSchedule));
    o.learning_rate               = w.learningRate;
    o.warmup_steps                = w.warmupSteps;
    o.lr_cooling_factor           = w.lrCoolingFactor;
    o.lr_decay_factor             = w.lrDecayFactor;
    o.lr_decay_threshold          = w.lrDecayThreshold;
    o.lr_adapt_patience           = w.lrAdaptPatience;
    o.lr_growth_factor            = w.lrGrowthFactor;
    o.lr_growth_threshold         = w.lrGrowthThreshold;
    o.stop_criterion              = static_cast<StopCriterion>(static_cast<int32_t>(w.stopCriterion));
    o.stop_displacement_tol       = w.stopDisplacementTol;
    o.stop_displacement_patience  = w.stopDisplacementPatience;
    o.stop_loss_tol               = w.stopLossTol;
    o.stop_loss_patience          = w.stopLossPatience;
    return o;
}

// ---------- Embedder construction ----------

std::unique_ptr<wembed::Embedder> create_embedder(const wembed::Graph& g, const Options& opts) {
    return std::make_unique<wembed::Embedder>(wembed::createEmbedder(g, to_wembed(opts)));
}

// ---------- Embedder accessors ----------

void embedder_copy_coords(const wembed::Embedder& e, rust::Slice<double> out) {
    const std::size_t expected = static_cast<std::size_t>(e.getNumVertices())
                                * static_cast<std::size_t>(e.getEmbeddingDimension());
    if (out.size() != expected) {
        throw std::invalid_argument(
            "embedder_copy_coords: out buffer size mismatch (got "
            + std::to_string(out.size()) + ", expected " + std::to_string(expected) + ")");
    }
    e.copyCoordinatesTo(out.data());
}

std::unique_ptr<std::vector<double>> embedder_weights(const wembed::Embedder& e) {
    return std::make_unique<std::vector<double>>(e.getWeights());
}

std::unique_ptr<wembed::Graph> embedder_current_graph(const wembed::Embedder& e) {
    return std::make_unique<wembed::Graph>(e.getCurrentGraph());
}

rust::String embedder_timings_str(const wembed::Embedder& e) {
    return rust::String(wembed::timingsToString(e.getTimings()));
}

Loss embedder_loss(const wembed::Embedder& e) {
    const wembed::Loss l = e.getLoss();
    Loss out;
    out.attractive = l.attractive;
    out.repulsive  = l.repulsive;
    out.total      = l.total;
    return out;
}

Progress embedder_progress(const wembed::Embedder& e) {
    const wembed::Progress p = e.getProgress();
    Progress out;
    out.layer               = p.layer;
    out.num_layers          = p.numLayers;
    out.num_vertices        = p.numVertices;
    out.iteration           = p.iteration;
    out.expected_iterations = p.expectedIterations;
    out.layer_seconds       = p.layerSeconds;
    out.eta_seconds         = p.etaSeconds;
    out.layer_finished      = p.layerFinished;
    return out;
}

// ---------- Free helpers ----------

void set_seed(int32_t seed) {
    wembed::setSeed(seed);
}

}  // namespace wembed_sys
