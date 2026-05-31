# cuda/ — MPS tensor-network quantum-annealing engine (C++)

The tensor-network pillar of TESSERA. A matrix-product-state engine evolves the
many-body wavefunction |ψ⟩ under the time-dependent transverse-field Ising
Hamiltonian via second-order TEBD, breaking the 2ⁿ state-vector wall.

```
H(s) = -A(s)·Σ_i X_i  -  B(s)·[ Σ_<i,j> J_ij Z_i Z_j + Σ_i h_i Z_i ]
```

The bond dimension **χ** caps representable entanglement: at **χ ≥ 2^(n/2)** the
simulation is exact (matches the state-vector oracle in `rust/`); below that it
is a controlled approximation whose discarded Schmidt weight is reported.

## components (pure C++17, self-contained — no external linalg deps)

| file | role |
|------|------|
| `include/linalg.hpp`, `src/linalg.cpp` | complex Hermitian eigendecomposition (real 2n embedding + cyclic Jacobi), Hermitian matrix exponential, thin SVD via the smaller Gram matrix |
| `include/mps.hpp`, `src/mps.cpp` | MPS state, single-/two-site gates with SVD bond truncation, long-range Z_iZ_j via swap networks, observables (⟨Z⟩, ⟨ZZ⟩, ⟨H⟩, amplitude prob), single-site entanglement entropy, the `mps_anneal` TEBD driver |
| `src/ffi.cpp` | C ABI (`tessera_qa_anneal`) consumed by `rust/` under `--features cuda`; CMake builds it as the shared lib `tessera_qa` |
| `tests/test_linalg.cpp`, `tests/test_mps.cpp` | ctest suites |

## build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build            # or run build/test_linalg, build/test_mps
```

## status — honest scope

**Verified working:**
- linalg: Pauli spectra, exp unitarity, V·diag(w)·V† reconstruction (5/5).
- MPS represents states with discarded weight ~1e-30 at exact χ.
- ⟨Z⟩, ⟨ZZ⟩ measurement validated on product states.
- Reaches the **exact ground state** on n≤6 random instances (incl. long-range
  couplings) and on a **frustrated 3-spin ring** (⟨H⟩ converges to exact as the
  adiabatic time grows).
- Entanglement entropy peaks mid-anneal (the quantum phase transition).
- Linked into `rust/` via `--features cuda`; the `bench` QA(mps) column is live.

**Known limitation (tracked, not hidden):** dense long-range graphs at n≥8 do
not yet reach the ground state with the current fixed cos²/sin² schedule — ⟨H⟩
stalls in an excited manifold even though the representation stays exact
(trunc≈0). This is an adiabatic-path / schedule problem (gap structure), not a
tensor-network bug. Fix in progress: instance-aware schedule (slow-down near the
minimum gap) and/or GNN-predicted schedules (Phase 2). The `test_mps` suite
prints this case as `[KNOWN_LIMIT]` rather than asserting it works.

## next

- cuSOLVER batched-SVD GPU layer on this validated CPU core (the "cuda" in the
  directory name — current code is the correct CPU reference the GPU port targets).
- Adaptive adiabatic schedule for dense graphs.
