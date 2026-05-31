// TESSERA — Matrix Product State engine for real quantum annealing.
//
// Represents the many-body wavefunction |ψ⟩ of n qubits as a chain of rank-3
// tensors (matrix product state) and evolves it under the time-dependent
// transverse-field Ising Hamiltonian via TEBD (Trotter + SVD truncation):
//
//   H(s) = -A(s)·Σ_i X_i  -  B(s)·[ Σ_<i,j> J_ij Z_i Z_j + Σ_i h_i Z_i ]
//
// The MPS bond dimension χ caps the representable entanglement: when χ ≥ 2^(n/2)
// the simulation is EXACT (matches the state-vector oracle); below that it is a
// controlled approximation whose discarded weight we report. This is what breaks
// the 2^n memory wall — the whole point of tensor networks.
//
// Long-range Z_iZ_j couplings (arbitrary graphs) are applied via swap networks,
// so the engine is general, not just nearest-neighbour. The honest cost of a
// non-1D problem shows up as a larger required χ (more entanglement).
//
// Pure C++ / std::complex, no external deps. The cuSOLVER batched-SVD GPU layer
// mounts on top of this validated CPU core.
#pragma once
#include "linalg.hpp"
#include <vector>
#include <cstdint>

namespace tessera {

// One MPS tensor: shape (Dl, 2, Dr). element (a, s, b) at (a*2 + s)*Dr + b.
struct Site {
    int Dl = 1, d = 2, Dr = 1;
    std::vector<cd> t;
    Site() : t(2, cd(0, 0)) {}
    Site(int dl, int dr) : Dl(dl), Dr(dr), t((std::size_t)dl * 2 * dr, cd(0, 0)) {}
    cd& operator()(int a, int s, int b) { return t[((std::size_t)a * 2 + s) * Dr + b]; }
    cd operator()(int a, int s, int b) const { return t[((std::size_t)a * 2 + s) * Dr + b]; }
};

struct AnnealResult {
    std::vector<int8_t> spins;     // ±1 readout (sign of ⟨Z_i⟩)
    double energy = 0.0;           // Ising energy of `spins`
    double expected_energy = 0.0;  // <psi|H|psi> measured from the MPS (no rounding)
    double prob = 0.0;             // |amplitude|² of the readout configuration
    double max_trunc = 0.0;        // largest discarded Schmidt weight over the run
    std::vector<double> entropy;   // single-site (qubit 0) entanglement entropy per step
};

class MPS {
public:
    explicit MPS(int n);                       // |+⟩^n product state
    int size() const { return n_; }

    // Single-site gate G (2×2) applied to qubit i.
    void apply_single(int i, const cd G[2][2]);
    // Two-site gate G (4×4, basis |s_i s_{i+1}⟩) on the adjacent pair (i,i+1),
    // SVD-truncated to bond dimension `chi`. Accumulates discarded weight.
    void apply_adjacent(int i, const cd G[4][4], int chi, double& trunc);
    // Z_iZ_j coupling gate exp(-i·θ·Z_iZ_j) for arbitrary i,j via swap network.
    void apply_zz(int i, int j, double theta, int chi, double& trunc);

    double norm2() const;                       // ⟨ψ|ψ⟩
    double expect_z(int i) const;               // ⟨Z_i⟩ / ⟨ψ|ψ⟩
    double expect_zz(int i, int j) const;       // ⟨Z_iZ_j⟩ / ⟨ψ|ψ⟩
    double single_site_entropy(int i) const;    // vN entropy of qubit i (bits)
    double amplitude_prob(const std::vector<int8_t>& spins) const;  // |⟨x|ψ⟩|²/⟨ψ|ψ⟩
    int max_bond() const;

private:
    int n_;
    std::vector<Site> s_;
    // expectation of Π_{k in zsites} Z_k (zsites sorted) over the MPS, unnormalised.
    cd expect_zstring_raw(const std::vector<int>& zsites) const;
};

// Run real quantum annealing via MPS-TEBD. `ei,ej,ew` list the Ising couplings
// (arbitrary graph). `sa,sb` are the discretised A(s),B(s) schedules (len steps).
// Writes the ±1 readout into out_spins; entropy trace (len steps) if non-null.
AnnealResult mps_anneal(const std::vector<double>& h,
                        const std::vector<int>& ei,
                        const std::vector<int>& ej,
                        const std::vector<double>& ew,
                        const std::vector<double>& sa,
                        const std::vector<double>& sb,
                        double dt, int chi, int zz_substeps = 1);

}  // namespace tessera
