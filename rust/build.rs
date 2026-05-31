// Build script. Only active under `--features cuda`: tells the linker where to
// find the C++ MPS engine import library (cuda/build/tessera_qa.lib) using an
// absolute path derived from the crate dir, which is more robust than relative
// -L flags in .cargo/config.toml for cdylib targets.
//
// Default build (no `cuda` feature) does nothing, so the crate stays pure-Rust
// and needs no C++ toolkit / GPU.

use std::path::PathBuf;

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    // crate dir = rust/ ; engine builds to ../cuda/build
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = manifest.parent().unwrap().join("cuda").join("build");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=tessera_qa");
    println!("cargo:rerun-if-changed=../cuda/src/ffi.cpp");
    println!("cargo:rerun-if-changed=build.rs");
}
