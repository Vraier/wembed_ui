// Thin cxx shim over wembed's public C++ API.
//
// Every function here exists to adapt a signature that cxx cannot represent directly
// (return-by-value opaque types, nested std::vectors, std::string fields). We deliberately
// keep the shim narrow — no logic, no state, no caching. If a wembed API is already
// cxx-friendly, we call it directly from the bridge and it does not appear here.

#pragma once
 
#include <memory>
#include <string>

#include "rust/cxx.h"
#include "wembed.h"

// cxx generates the shared types (Edge, Options, SpatialIndex) inside `namespace wembed_sys`,
// derived from the bridge's `namespace = "wembed_sys"`. Our free-function shims must live in
// the same namespace so cxx's generated wrappers can find them by qualified name.

namespace wembed_sys {

// Forward decls of the cxx-generated shared types; full definitions come from
// "wembed-sys/src/lib.rs.h", included at implementation time in shim.cc.
struct Edge;
struct Options;
struct Loss;
struct Progress;
enum class SpatialIndex : ::std::int32_t;

// ---- Graph construction ----
std::unique_ptr<wembed::Graph> graph_from_edges(rust::Slice<const Edge> edges);
std::unique_ptr<wembed::Graph> graph_from_file(rust::Str path, rust::Str comment, rust::Str delimiter);

// ---- Graph accessors ----
std::unique_ptr<std::vector<Edge>> graph_edge_list(const wembed::Graph& g);

// ---- Options defaults ----
// Default-constructed wembed::Options, so Rust's `Options::default()` can source the
// values from the C++ header (the single source of truth) instead of hand-copying them.
Options default_options();

// ---- Embedder construction ----
std::unique_ptr<wembed::Embedder> create_embedder(const wembed::Graph& g, const Options& opts);

// ---- Embedder accessors ----
void                                 embedder_copy_coords(const wembed::Embedder& e, rust::Slice<double> out);
std::unique_ptr<std::vector<double>> embedder_weights(const wembed::Embedder& e);
std::unique_ptr<wembed::Graph>       embedder_current_graph(const wembed::Embedder& e);
rust::String                         embedder_timings_str(const wembed::Embedder& e);
Loss                                 embedder_loss(const wembed::Embedder& e);
Progress                             embedder_progress(const wembed::Embedder& e);

// ---- Free helpers ----
void set_seed(int32_t seed);

}  // namespace wembed_sys
