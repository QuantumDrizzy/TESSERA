# ADR-0001: TESSERA — Neural-Guided Real Quantum Annealing via Tensor Networks

**Status:** Proposed
**Date:** 2026-05-30
**Deciders:** Antonio

---

## Context

The portfolio needs a **flagship** — one differential project that unifies the scattered engines (GNN, tensor networks, annealing) into a single coherent story, instead of 11 separate repos. It must:

- combine **GNN + tensor networks + quantum annealing**,
- use **AI + real quantum dynamics** (not a quantum-inspired heuristic),
- run **bare-metal, local, sovereign** — no cloud, no QPU, no web, no localhost,
- be written in **Rust + C/C++/CUDA + Python**, each where it fits best,
- be **honest and benchmarkable** (no "quantum advantage" claims it can't defend),
- be **distinct from SESHAT** — separate project, separate engine, no shared code.

The job target is salaried & remote (EU/Zürich); the most thematically aligned employer is **Terra Quantum** (tensor-network-based quantum computation + optimization + QML), so this project doubles as interview ammunition for exactly that profile.

## Decision

Build **TESSERA**: a local, neural-guided **quantum annealer** that solves QUBO/Ising optimization by simulating the **real quantum adiabatic process** — the time-dependent transverse-field Ising Hamiltonian — using **tensor networks (matrix product states + TDVP/TEBD)** on bare-metal GPU, with a **GNN that learns to guide the anneal**.

> **The unifying idea — three views of one graph.**
> A QUBO/Ising problem *is* a graph (spins = nodes, couplings `J_ij` = edges).
> - **GNN → LEARNS** the graph (predicts a good schedule / warm-start).
> - **Tensor network → REPRESENTS** the graph's many-body quantum state.
> - **Quantum annealing → SEARCHES** the graph's ground state via adiabatic evolution.
>
> They are not bolted together; they are three operations on the same object.

### The pipeline

```
   problem  ──►  QUBO / Ising graph  (h_i, J_ij)
                        │
        ┌───────────────┴────────────────┐
        ▼                                 │
  [1] GNN  (Python / AI)                  │
      reads the graph, predicts an        │
      annealing schedule A(s),B(s)        │
      and a warm-start product state      │
        │ amortized: train once, solve many
        ▼                                 │
  [2] QUANTUM ANNEALING  (real adiabatic dynamics)        ◄── this is what makes it
      H(s) = -A(s)·Σ σ^x  -  B(s)·[ Σ J σ^z σ^z + Σ h σ^z ]    "real quantum", and
      evolve |ψ⟩ from the transverse-field ground state          DIFFERENT from SESHAT
      to the problem Hamiltonian (tunneling, superposition)      (which is classical SA)
        │
        ▼
  [3] TENSOR NETWORK ENGINE  (C++/CUDA)
      |ψ⟩ as an MPS, evolved by TDVP/TEBD — the substrate that
      makes simulating the real quantum evolution tractable on
      bare metal (no exponential blowup, no QPU, no cloud)
        │
        ▼
   readout: sample the MPS  ──►  bitstring  ──►  solution
   observables: entanglement entropy + energy gap along the anneal
```

## Distinction from SESHAT (explicit, by request)

| | SESHAT | TESSERA |
|---|--------|---------|
| Annealing type | **Classical** simulated annealing (thermal, Metropolis) | **Quantum** annealing (adiabatic, real Schrödinger evolution) |
| Mechanism | Temperature schedule, random thermal hops | Transverse-field → problem Hamiltonian, quantum tunneling |
| Engine | Spin-flip SA | MPS + TDVP/TEBD tensor-network simulation |
| Domain | Decipherment / QUBO | General optimization + neural guidance |
| Code sharing | — | **None.** TESSERA has its own independent SA *baseline* purely so quantum-vs-classical can be benchmarked honestly. |

They are two separate repos. Their only conceptual link is the honest benchmark axis (quantum vs classical on identical instances).

## Honesty — what "real quantum" means here (stated precisely)

This is the project's signature and must be airtight:

- TESSERA **numerically integrates the time-dependent Schrödinger equation** for a transverse-field Ising Hamiltonian, representing the many-body wavefunction as an MPS and evolving it with TDVP/TEBD. **This is the same physics a hardware quantum annealer realizes** — entanglement, superposition, and tunneling are all present and *measurable*.
- It is a **faithful classical simulation of real quantum dynamics** — **not** a "quantum-inspired" classical heuristic, and **not** a physical QPU.
- **Strength a QPU can't match:** full-state introspection. We can plot the **entanglement entropy spike at the quantum phase transition** during the anneal — a real D-Wave can't show you that.
- **Honest limits:** an MPS is exact only up to **bond dimension χ**. High-entanglement instances are approximated — we **always report χ and the truncation error**, and validate against the exact brute-force ground state on small `n`.
- **Optional real-QPU backend** (D-Wave / IBM) is a pluggable *validation* path, **OFF by default** (cloud would break sovereignty). Local-first, always.

## Architecture / languages (each where it fits — bare metal)

```
Python   GNN (PyTorch + PyTorch-Geometric): learns schedule & warm-start.
         Training, benchmark harness, plotting, high-level API.          ← AI / ML (favourite turf)
   │  (schedule + warm-start)         ▲ (results, observables)
   ▼  PyO3? no — ctypes/CLI            │
Rust     Orchestration: problem model, adiabatic-schedule driver,
         INDEPENDENT classical SA baseline, exact solver (small n),
         benchmark harness, safe FFI to the CUDA engine, CLI.            ← systems backbone (safety)
   │  FFI (C ABI)                      ▲
   ▼                                   │
C++/CUDA Tensor-network engine: MPS state, SVD/QR, MPO application,
         TEBD/TDVP time evolution of the transverse-field Ising.         ← REAL QUANTUM hot core
         The perf-critical many-body simulation, on Blackwell.
```

No web languages. No localhost. No cloud. Fully local & sovereign.

## Options Considered

### Option A: Reuse QUENCH (webcam vision) as the flagship
| Dimension | Assessment |
|-----------|------------|
| Differential | Low — a fun demo, but a toy |
| Scope fit | Narrow (segmentation only) |
**Rejected.** QUENCH stays a downstream *demo application* (vision segmentation can be one of TESSERA's example problems), not the flagship.

### Option B: Classical "quantum-inspired" SA only
**Rejected.** Not real quantum, not differential, and overlaps SESHAT.

### Option C: Real quantum annealing via tensor networks, neural-guided  ✅ CHOSEN
| Dimension | Assessment |
|-----------|------------|
| Differential | **High** — local real-QA-via-TN + neural guidance is genuinely rare |
| Terra-Quantum fit | **Exact** — their core thesis |
| Reuses skills | TDVP/MPS (SUBSTRATE), GNN (cycle_project/LHCb), CUDA (Blackwell) |
| Honesty | Strong — full-state observables + reported χ/truncation |

### Option D: Real-QPU only (D-Wave / IBM cloud)
**Rejected.** Violates sovereignty / local / no-cloud. Kept as an *optional* validation backend.

## Trade-off Analysis

- **TN simulation cost vs QPU:** TN scales with entanglement (bond dimension χ), not naively with `2^n`. Excellent for low/moderate-entanglement and 1D-like couplings; expensive for highly-connected, high-entanglement 2D instances → roadmap PEPS later. Honest scope: shine where MPS is appropriate, report where it isn't.
- **GNN amortization:** training a GNN to predict schedules/warm-starts pays off only when solving *many* instances from a distribution. For one-off problems it's overhead. Benchmark the amortized regime explicitly.
- **Real quantum vs classical:** quantum annealing isn't universally faster than SA — the honest contribution is *characterizing* where adiabatic/tunnelling beats thermal hopping (e.g., tall-thin energy barriers), measured on identical instances.

## Consequences

**Easier:**
- One coherent flagship narrative ("neural-guided real quantum annealing, local & sovereign") replacing scattered repos → findable.
- Honest quantum-vs-classical benchmarks on identical instances (TESSERA-SA vs TESSERA-QA vs exact).
- Full quantum-state observability (entanglement entropy, energy gap) — great for demos & interviews.

**Harder:**
- The CUDA tensor-network engine is the real work: stable TDVP, GPU SVD/QR, truncation control.
- GNN needs an instance-distribution + training loop; schedule parameterization must stay physically valid (monotone A↓, B↑).
- Bond-dimension / memory management on large instances.

**Revisit later:**
- PEUS/PEPS for 2D high-entanglement problems.
- Optional real-QPU validation backend.
- QUENCH as a live vision demo driven by the TESSERA engine.

## Build Sequence (phased — each phase independently demoable)

- [ ] **Phase 0 — Foundations (Rust, verifiable now).** Ising/QUBO model, independent classical SA baseline, exact brute-force ground state (small `n`), benchmark harness. *Gives baselines + ground truth before any quantum code.* ← seeded in this commit, `cargo test` green.
- [~] **Phase 1 — Real quantum core.** Two backends, one physics:
  - [x] **Exact state-vector oracle (Rust, `quantum.rs`).** Real-time adiabatic evolution of the transverse-field Ising Hamiltonian on the full 2ⁿ amplitudes via symmetric Trotter. Exact quantum dynamics (no truncation) for small `n` — both a working quantum annealer and the ground-truth oracle for the GPU engine. Validated against `exact_ground_state`.
  - [ ] **Scalable MPS engine (C++/CUDA + Rust FFI).** MPS state + TEBD/TDVP evolution; Rust schedule driver; MPS sampling for readout; validate vs the state-vector oracle on small `n`, then scale past 2ⁿ memory limits.
- [ ] **Phase 2 — Neural guidance (Python).** GNN (PyG) predicts annealing schedule + warm-start from the problem graph; amortized training over an instance distribution; benchmark the quality/time gain honestly.
- [ ] **Phase 3 — Observables & demos.** Entanglement-entropy & energy-gap tracking along the anneal + visualization; optional QUENCH-vision demo; optional real-QPU validation backend.
- [ ] **Phase 4 — Polish.** CI (cargo + pytest + clippy), docs, **honest benchmark report**, write-up / blog → findable & publishable.

## Action Items

1. [ ] Scaffold repo (`rust/`, `cuda/`, `python/`, `docs/`, `benchmarks/`).
2. [ ] Phase 0 Rust: `Ising`, `simulated_annealing` (baseline), `exact_ground_state`, tests.
3. [ ] Define the C-ABI for the CUDA TN engine (`tessera_qa_anneal(...)`) before writing CUDA.
4. [ ] Phase 1 CUDA: MPS + TEBD transverse-field Ising; Rust FFI; validate vs exact.
5. [ ] Phase 2 Python: GNN schedule/warm-start; training; benchmark.
6. [ ] Honest benchmark report: TESSERA-QA vs TESSERA-SA vs exact (report χ, truncation, timings).
