# TESSERA

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

- ✅ **Phase 0 (Rust foundations):** Ising/QUBO model + QUBO⇄Ising conversion + classical SA baseline + exact ground-state solver + adiabatic schedules + benchmark harness.
- 🟡 **Phase 1 (real quantum annealing):**
  - ✅ **exact state-vector oracle** (`quantum.rs`) — real-time adiabatic evolution of the transverse-field Ising Hamiltonian on the full 2ⁿ amplitude vector (symmetric Trotter). Real quantum dynamics for small n; the ground-truth reference the CUDA engine is validated against.
  - ⏳ **scalable CUDA MPS engine** (`cuda/`) — TEBD/TDVP tensor-network evolution on GPU, behind the `cuda` feature.
- 🟡 **Phase 3 (fused webcam demo):** live segmentation as Ising ground state, solved by **classical SA or real quantum annealing**, toggled live. Python vision pipeline (`python/tessera_cam.py`) calls the Rust solvers via a C ABI (ctypes). FFI round-trip verified (`python/test_ffi.py` → PASS); needs a camera + `opencv-contrib-python` to run live. Quantum path is exact `2ⁿ`, so frames are coarsened to ≤ ~22 regions.
- ⏳ Phase 2: GNN neural guidance.
- ⏳ Phase 1b / future: CUDA MPS engine for full-resolution quantum; entanglement-entropy observables.

## build

```bash
cd rust && cargo test          # model + baselines + quantum oracle vs exact ground states
cargo run --bin bench          # exact vs classical SA vs real quantum annealing (state vector)
```

Two quantum backends, one physics: `quantum.rs` (exact 2ⁿ state vector, CPU — the oracle) and `cuda/` (MPS+TDVP, GPU — scalable, validated against the oracle).
