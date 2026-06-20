#!/usr/bin/env sh
# TESSERA — push & play: build the Rust solver core, then run the live webcam demo.
#
#   git clone https://github.com/QuantumDrizzy/TESSERA && cd TESSERA && sh run.sh
#
# Prereqs: a Rust toolchain (cargo), Python 3, a webcam, and the demo deps:
#   pip install -r python/requirements-demo.txt
#
# The Rust core is pure Rust (no CUDA/GPU) — it builds anywhere, incl. aarch64
# (Raspberry Pi). The optional C++/CUDA MPS engine is a separate, off-by-default
# path (see cuda/README.md); the live demo does not need it.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/2] Building the Rust solver core (release, pure Rust — no CUDA)..."
( cd "$DIR/rust" && cargo build --release )

echo "[2/2] Launching the webcam demo  -  q: quit | m: SA<->QA | +/-: QA regions"
( cd "$DIR/python" && python tessera_cam.py )
