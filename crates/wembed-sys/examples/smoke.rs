//! End-to-end FFI proof: load the wembed test graph, run 10 steps, print coords.
//!
//! Compare against `build/bin/wembed -i assets/small_graph.edg -o /tmp/smoke.emb --dim 2`
//! in the wembed repo. Values should agree within floating-point tolerance for the same seed.

use std::env;
use std::path::PathBuf;

use wembed_sys::{ffi, Options, SpatialIndex};

fn main() {
    let wembed_src = env::var_os("WEMBED_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("third_party")
                .join("wembed")
        });
    let graph_path = wembed_src.join("assets").join("small_graph.edg");
    assert!(
        graph_path.exists(),
        "expected test graph at {}",
        graph_path.display()
    );

    ffi::set_seed(42);

    let graph = ffi::graph_from_file(graph_path.to_str().unwrap(), "#", " ").expect("load graph");
    let n = graph.num_vertices() as usize;
    println!("graph: {} vertices, {} edges", n, graph.num_edges());

    let opts = Options {
        embedding_dimension: 2,
        max_iterations: 10,
        index_type: SpatialIndex::IndexSprk,
        ..Options::default()
    };
    let dim = opts.embedding_dimension as usize;

    let mut embedder = ffi::create_embedder(&graph, &opts).expect("create embedder");

    // Sanity: cheap accessors agree with what we know.
    assert_eq!(embedder.num_vertices() as usize, n);
    assert_eq!(embedder.embedding_dimension() as usize, dim);

    for step in 0..10 {
        if embedder.is_finished() {
            println!("converged at step {}", step);
            break;
        }
        embedder.pin_mut().calculate_step();
    }

    println!("progress: {:?}", ffi::embedder_progress(&embedder));

    // Zero-alloc readout into a caller-owned buffer.
    let mut flat = vec![0.0_f64; n * dim];
    ffi::embedder_copy_coords(&embedder, &mut flat);

    println!("coords ({} × {}):", n, dim);
    for v in 0..n {
        println!("  v{}: {:?}", v, &flat[v * dim..(v + 1) * dim]);
    }

    println!("\ntimings:\n{}", ffi::embedder_timings_str(&embedder));
}
