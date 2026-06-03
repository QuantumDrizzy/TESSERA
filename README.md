# TESSERA

[![CI](https://github.com/QuantumDrizzy/TESSERA/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantumDrizzy/TESSERA/actions/workflows/ci.yml)
![Rust](https://img.shields.io/badge/Rust-systems%20core-orange)
![C++](https://img.shields.io/badge/C%2B%2B-MPS%20engine-blue)
![Python](https://img.shields.io/badge/Python-GNN%20guidance-yellow)

**Neural-guided real quantum annealing via tensor networks** — bare metal, local, sovereign. No cloud, no QPU, no web.

TESSERA solves QUBO/Ising optimization by simulating the **real quantum adiabatic process** (transverse-field Ising — tunneling, superposition) with **tensor networks** (MPS + TDVP/TEBD) on a local GPU, while a **GNN learns to guide the anneal**.

> **One graph, three views.** A QUBO/Ising problem is a graph:
> **GNN learns it · tensor network represents its quantum state · quantum annealing searches its ground state.**

## why it's different

- **Real quantum, not "quantum-inspired".** TESSERA integrates the time-dependent Schrödinger equation for a transverse-field Ising Hamiltonian as a matrix product state — the same physics a hardware annealer realizes — and lets you *watch the entanglement entropy spike at the quantum phase transition*, which no real QPU can show you.
- **Local & sovereign.** Runs entirely on your own GPU. No cloud, no QPU rental. An optional real-QPU validation backend exists but is off by default.
- **Honest.** MPS is exact only up to bond dimension χ — TESSERA always reports χ and the truncation error, and validates against the exact ground state on small instances.
- **Distinct from [SESHAT](https://github.com/QuantumDrizzy/SESHAT)** (which is *classical* simulated annealing). Different physics, different engine, separate repo.

## architecture (each language where it fits)

| layer | language | role |
|-------|----------|------|
| `python/` | **Python** (PyTorch + PyG) | GNN that learns the annealing schedule & warm-start; training; benchmarks; API |
| `rust/`   | **Rust** | problem model, adiabatic-schedule driver, **independent** classical SA baseline, exact solver, FFI, CLI |
| `cuda/`   | **C++/CUDA** | tensor-network engine: MPS + TEBD/TDVP real quantum adiabatic evolution (the hot core) |

See [`docs/ADR-0001-tessera.md`](docs/ADR-0001-tessera.md) for the full design.

## status

**All three pillars are implemented and tested — one Ising graph, three views: the GNN learns it, the tensor network represents its quantum state, quantum annealing searches its ground state.**

- ✅ **Phase 0 (Rust foundations):** Ising/QUBO model + QUBO⇄Ising conversion + classical SA baseline + exact ground-state solver + adiabatic schedules + benchmark harness.
- ✅ **Phase 1a (real quantum annealing):** exact state-vector oracle (`rust/src/quantum.rs`) — real-time adiabatic evolution of the transverse-field Ising Hamiltonian on the full 2ⁿ amplitude vector (symmetric Trotter). Real quantum dynamics for small n; the ground-truth reference the MPS engine is validated against. Reports the live entanglement-entropy trace (the quantum phase transition).
- ✅ **Phase 1b (tensor networks):** MPS-TEBD quantum-annealing engine in C++ (`cuda/`), breaking the 2ⁿ wall. χ caps representable entanglement (exact at χ ≥ 2^(n/2), reported discarded weight below). **Verified:** trunc ~1e-30 at exact χ; reaches the exact ground state on n≤6 random instances (incl. long-range couplings) and a frustrated 3-spin ring; linalg 5/5 + MPS suite pass. Linked into Rust via `--features cuda`. *Known limit (documented, not hidden): dense graphs at n≥8 stall the fixed schedule — an adiabatic-path issue, not a TN bug.*
- ✅ **Phase 2 (GNN guidance):** physics-informed graph neural network (`python/gnn_guide.py`, pure-torch). Reads the Ising graph, predicts a per-spin warm-start, trained **unsupervised with the Ising energy as the loss** (no labels, no solver in the loop). **Verified:** relaxed energy +0.01 → -7.31 while training; on unseen instances 0.21 mean gap to optimum vs 0.53 for best-of-10 random, beating random 24/30.
- ✅ **Phase 3 (fused webcam demo):** live segmentation as Ising ground state, solved by **classical SA or real quantum annealing**, toggled live (`python/tessera_cam.py` → Rust solvers via ctypes). Live entanglement-entropy panel. Needs a camera + `opencv-contrib-python`.

## build

```bash
# Rust core (oracle + classical baselines + bench)
cd rust && cargo test
cargo run --release --bin bench

# C++ MPS tensor-network engine
cd cuda && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build

# Rust + MPS engine together (fills the QA(mps) column)
cd rust && cargo run --release --features cuda --bin bench   # copy cuda/build/tessera_qa.dll next to the exe

# GNN guidance (Phase 2)
cd python && python test_gnn.py        # or: python gnn_guide.py --n 12
```

Two quantum backends, one physics: `quantum.rs` (exact 2ⁿ state vector, CPU — the oracle) and `cuda/` (MPS-TEBD — scalable, validated against the oracle). The cuSOLVER GPU layer mounts on the validated CPU MPS core.
