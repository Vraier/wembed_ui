{
  description = "wembed_ui - interactive viewer for the wembed graph embedder";

  inputs = {
    # wembed itself is the third_party/wembed submodule (needs Nix >= 2.27)
    self.submodules = true;
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # wembed's CMake would download sprk; the build sandbox has no network. Keep in sync
    # with third_party/wembed/src/sprk/CMakeLists.txt (and the sprk rev in Cargo.toml).
    sprk = {
      url = "github:wembed-pdf/sprk/1e195eab1119aa5cf420bfb322788d53b95bf1ad";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    sprk,
  }:
    # build.rs links GCC's OpenMP + libstdc++
    flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux"] (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;

        # dlopen'd by winit / wgpu at runtime
        runtimeLibs = with pkgs; [
          vulkan-loader
          wayland
          libxkbcommon
          libx11
          libxcursor
          libxi
          libxrandr
        ];

        wembed-viewer = pkgs.rustPlatform.buildRustPackage {
          pname = "wembed-viewer";
          version = "0.1.0";
          src = self;

          # one vendor dir for both cargo builds: the viewer's and the nested one wembed's
          # CMake (Corrosion) runs for sprk-ffi (see preBuild)
          cargoDeps = pkgs.symlinkJoin {
            # importCargoLock's config.toml points at a dir of exactly this name
            name = "cargo-vendor-dir";
            paths = [
              (pkgs.rustPlatform.importCargoLock {
                lockFile = ./Cargo.lock;
                outputHashes."sprk-0.2.0" = "sha256-f5KJ5FtwIbiyN8qWkdpX6lzJNaCCOh+1ECBLZzI/1Eg=";
              })
              (pkgs.rustPlatform.importCargoLock {lockFile = "${sprk}/Cargo.lock";})
            ];
          };

          nativeBuildInputs = with pkgs; [cmake ninja];
          # build.rs drives CMake/Ninja itself
          dontUseCmakeConfigure = true;
          dontUseNinjaBuild = true;
          dontUseNinjaInstall = true;
          dontUseNinjaCheck = true;

          # Corrosion runs the nested cargo from sprk's store dir, outside the source tree, so
          # it misses the source-local .cargo/config.toml; CARGO_HOME's config reaches it
          preBuild = ''
            export CARGO_HOME=$TMPDIR/cargo-home
            mkdir -p $CARGO_HOME
            cat > $CARGO_HOME/config.toml <<EOF
            [source.crates-io]
            replace-with = "vendored-sources"
            [source.vendored-sources]
            directory = "$NIX_BUILD_TOP/cargo-vendor-dir"
            EOF
          '';

          env = {
            FETCHCONTENT_FULLY_DISCONNECTED = "ON";
            FETCHCONTENT_SOURCE_DIR_CORROSION = "${pkgs.corrosion.src}";
            FETCHCONTENT_SOURCE_DIR_SPRK = "${sprk}";
            WEMBED_VIEWER_THEMES_DIR = "${placeholder "out"}/share/wembed-viewer/themes";
          };

          postInstall = ''
            mkdir -p $out/share/wembed-viewer
            cp -r crates/wembed-viewer/themes $out/share/wembed-viewer/
          '';
          postFixup = ''
            patchelf --add-rpath ${lib.makeLibraryPath runtimeLibs} $out/bin/wembed-viewer
          '';

          meta = {
            description = "Live viewer and still renderer for wembed graph embeddings";
            homepage = "https://github.com/Vraier/wembed_ui";
            license = lib.licenses.mit;
            mainProgram = "wembed-viewer";
            platforms = lib.platforms.linux;
          };
        };
      in {
        packages.default = wembed-viewer;

        checks = {
          # build + cargo test (the GPU pixel test passes vacuously without an adapter)
          package = wembed-viewer;
          clippy = wembed-viewer.overrideAttrs (old: {
            pname = "wembed-viewer-clippy";
            nativeBuildInputs = old.nativeBuildInputs ++ [pkgs.clippy];
            buildPhase = "runHook preBuild; cargo clippy --workspace --all-targets -- -D warnings";
            doCheck = false;
            installPhase = "touch $out";
            postFixup = "";
          });
          fmt = pkgs.runCommand "wembed-ui-fmt" {nativeBuildInputs = [pkgs.cargo pkgs.rustfmt];} ''
            cd ${self}
            cargo fmt --all --check
            touch $out
          '';
        };

        devShells.default = pkgs.mkShell {
          # the wrapper re-adds -D_FORTIFY_SOURCE after user flags, which warns on every
          # unoptimized (debug) C++ compile of the shim
          hardeningDisable = ["fortify"];

          # libstdc++ needs to be reachable at runtime for anything cargo builds
          # that links against C++ (i.e. the wembed shim).
          LD_LIBRARY_PATH = lib.makeLibraryPath ([pkgs.stdenv.cc.cc.lib] ++ runtimeLibs);

          packages = with pkgs; [
            # Rust toolchain
            cargo
            rustc
            rustfmt
            clippy
            rust-analyzer

            # C++ build (for driving wembed's cmake from build.rs)
            cmake
            ninja
            pkg-config
            gcc
          ];

          shellHook = ''
            echo "wembed_ui dev shell ready."
          '';
        };
      }
    );
}
