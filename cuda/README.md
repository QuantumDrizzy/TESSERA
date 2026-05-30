# cuda/ — tensor-network quantum-annealing engine (C++/CUDA)

> Phase 1. The real-quantum hot core. Not yet implemented — this file specifies it.

Simulates the **real quantum adiabatic process** on bare-metal GPU:

```
H(s) = -A(s)·Σ_i σ^x_i  -  B(s)·[ Σ_<i,j> J_ij σ^z_i σ^z_j + Σ_i h_i σ^z_i ]
```

with `s : 0→1`, `A(0)≫B(0)` (transverse field dominates) → `A(1)≪B(1)` (problem Hamiltonian). The many-body wavefunction `|ψ(s)⟩` is a **matrix product state (MPS)**, evolved by **TEBD / TDVP**.

## planned components

| file | role |
|------|------|
| `mps.cu` / `mps.cuh` | MPS tensors, canonical form, GPU SVD/QR (cuSOLVER), bond-dimension truncation |
| `tebd.cu` | Trotterized time evolution under the time-dependent transverse-field Ising Hamiltonian |
| `observables.cu` | energy, magnetization, **entanglement entropy**, two-site correlations |
| `sample.cu` | sample a bitstring from the final MPS (the solution readout) |
| `ffi.cpp` | C ABI consumed by `rust/` |

## C ABI (draft — define before writing CUDA)

```c
// Anneal one Ising instance. Returns final energy; writes spins into out_s.
// schedule_a / schedule_b: discretized A(s), B(s) along the anneal (len `steps`).
// chi: max MPS bond dimension (the exactness/cost knob — always reported).
float tessera_qa_anneal(
    const double* h, size_t n,
    const size_t* ei, const size_t* ej, const double* ew, size_t m,
    const double* schedule_a, const double* schedule_b, size_t steps,
    size_t chi,
    int8_t* out_s,            // [n]
    double* out_entropy);     // [steps] entanglement entropy trace (optional, may be NULL)
```

## honesty

The MPS is **exact only up to bond dimension `chi`**. Every run reports `chi` and the
truncation error. Results are validated against `rust/`'s exact brute-force ground
state on small `n`. This is a faithful classical simulation of real quantum dynamics
— not a QPU, not a quantum-inspired heuristic.
