use std::env;
use std::path::PathBuf;

fn main() {
    // ---------- Locate the wembed source tree ----------
    // Default: the pinned submodule. WEMBED_SRC swaps in a local checkout.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let wembed_src: PathBuf = env::var_os("WEMBED_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../third_party/wembed"));
    assert!(
        wembed_src.join("include/wembed.h").exists(),
        "no include/wembed.h under {}. Run `git submodule update --init`, or point WEMBED_SRC \
         at a wembed checkout.",
        wembed_src.display()
    );
    let wembed_src = wembed_src.canonicalize().unwrap();
    let wembed_include = wembed_src.join("include");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ---------- Build wembed via cmake ----------
    // The `cmake` crate handles OUT_DIR management. Products land at
    // $OUT_DIR/build/lib/ (see wembed's CMAKE_ARCHIVE_OUTPUT_DIRECTORY).
    let mut cmake_cfg = cmake::Config::new(&wembed_src);
    cmake_cfg
        .define("WEMBED_BUILD_TOOLS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .generator("Ninja")
        .build_target("wembed")
        // own target dir for the cargo that Corrosion runs for sprk-ffi: an inherited
        // CARGO_TARGET_DIR is the outer build's, whose lock we hold → deadlock on configure
        .env("CARGO_TARGET_DIR", out_dir.join("sprk-target"))
        // empty = no wrapper: keeps `cargo clippy`'s clippy-driver (and its -D warnings)
        // off sprk-ffi, which isn't our code to lint
        .env("RUSTC_WORKSPACE_WRAPPER", "");
    // offline builds (the Nix package) point FetchContent at pre-fetched sources
    for (key, value) in env::vars().filter(|(k, _)| k.starts_with("FETCHCONTENT_")) {
        println!("cargo:rerun-if-env-changed={key}");
        cmake_cfg.define(&key, &value);
    }
    let dst = cmake_cfg.build();
    let lib_dir = dst.join("build").join("lib");
    for archive in ["libwembed.a", "libsprk_ffi.a"] {
        assert!(
            lib_dir.join(archive).exists(),
            "wembed build did not produce {archive} in {}",
            lib_dir.display()
        );
    }

    // ---------- Compile the cxx bridge + shim ----------
    cxx_build::bridge("src/lib.rs")
        .file("shim/shim.cc")
        .include("shim")
        .include(&wembed_include)
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-fopenmp")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("wembed_sys_shim");

    // ---------- Link the wembed static libs + system deps ----------
    // GNU ld resolves static archives left-to-right; each archive must appear before the
    // archives it depends on. Order: shim (implicit) → wembed → sprk_ffi → OpenMP
    // → C++ runtime.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib in ["wembed", "sprk_ffi"] {
        println!("cargo:rustc-link-lib=static={}", lib);
    }
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // ---------- Rebuild triggers ----------
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-changed=shim/shim.cc");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WEMBED_SRC");
    // whole trees, not just wembed.h, so edits in a local WEMBED_SRC checkout rebuild
    for path in ["CMakeLists.txt", "include", "src"] {
        println!("cargo:rerun-if-changed={}", wembed_src.join(path).display());
    }
}
