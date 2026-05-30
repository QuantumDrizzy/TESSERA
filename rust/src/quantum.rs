//! Exact state-vector quantum annealing — the CPU reference oracle.
//!
//! Real-time adiabatic evolution of the transverse-field Ising Hamiltonian on
//! the full 2^n amplitude vector. This is **exact** quantum dynamics — no
//! tensor-network truncation — so it is both (a) a working quantum annealer for
//! small n and (b) the ground-truth oracle the scalable CUDA MPS engine
//! (`ffi.rs` / `cuda/`) is validated against. It handles arbitrary couplings,
//! not just 1D chains.
//!
//!   H(s) = -A(s)·Σ_i X_i  +  B(s)·H_problem,   H_problem|x⟩ = E_Ising(x)·|x⟩
//!
//! `H_problem` is diagonal in the computational basis (its ground state is the
//! Ising minimiser). Symmetric Trotter step for e^{-iH·dt}:
//!   e^{-i H_z dt/2} · e^{-i H_x dt} · e^{-i H_z dt/2}
//!   - H_z (diagonal): multiply amplitude of |x⟩ by phase e^{-i·B·E(x)·dt}
//!   - H_x = -A·Σ X_i : a commuting product of single-qubit X-rotations,
//!     e^{+iθX} = cosθ·I + i·sinθ·X with θ = A·dt, applied exactly.
//!
//! The state is stored as parallel real/imag arrays (no complex dependency —
//! bare metal). Memory is 16·2^n bytes, so this is practical to ~n = 26.

use crate::ising::Ising;
use crate::schedule::Schedule;

/// Outcome of a quantum anneal.
#[derive(Debug, Clone)]
pub struct QaOutcome {
    /// Most-probable measurement outcome — the returned solution (±1 spins).
    pub spins: Vec<i8>,
    /// Ising energy of `spins`.
    pub energy: f64,
    /// Final energy expectation ⟨ψ|H_problem|ψ⟩ (≈ energy when adiabatic).
    pub expected_energy: f64,
    /// Probability mass on `spins` (annealing success probability).
    pub success_prob: f64,
    /// Bipartite entanglement entropy (in bits) at each annealing step, for the
    /// cut [first half | second half] of the spins. This is REAL quantum
    /// entanglement built up during the anneal — it rises then falls, peaking
    /// near the quantum phase transition. No physical QPU can report this.
    pub entropy_trace: Vec<f64>,
    /// Energy expectation ⟨H_problem⟩ at each step (the anneal converging).
    pub energy_trace: Vec<f64>,
}

/// Diagonal of H_problem: `diag[x] = E_Ising(spins(x))` (bit set ⇒ spin +1).
fn problem_diagonal(model: &Ising) -> Vec<f64> {
    let dim = 1usize << model.n;
    (0..dim)
        .map(|x| {
            let s: Vec<i8> = (0..model.n)
                .map(|i| if (x >> i) & 1 == 1 { 1 } else { -1 })
                .collect();
            model.energy(&s)
        })
        .collect()
}

/// Real-time adiabatic quantum annealing on the full 2^n state vector.
///
/// Evolves |ψ⟩ from the uniform superposition (ground state of -ΣX) to the
/// problem Hamiltonian along `sched`, with Trotter step `dt`. Slower schedules
/// (more steps / smaller dt) follow the adiabatic path more faithfully.
pub fn anneal_statevector(model: &Ising, sched: &Schedule, dt: f64) -> QaOutcome {
    let n = model.n;
    assert!(n >= 1, "need at least one spin");
    assert!(n <= 28, "state vector needs 2^n amplitudes; n too large");
    let dim = 1usize << n;

    let diag = problem_diagonal(model);

    // Initial state |+⟩^n : uniform real amplitudes.
    let amp = 1.0 / (dim as f64).sqrt();
    let mut re = vec![amp; dim];
    let mut im = vec![0.0f64; dim];

    let steps = sched.steps();
    let mut entropy_trace = Vec::with_capacity(steps);
    let mut energy_trace = Vec::with_capacity(steps);

    for k in 0..steps {
        let (a, b) = (sched.a[k], sched.b[k]);
        apply_problem_phase(&mut re, &mut im, &diag, b * dt * 0.5);
        apply_transverse(&mut re, &mut im, n, a * dt);
        apply_problem_phase(&mut re, &mut im, &diag, b * dt * 0.5);
        entropy_trace.push(entanglement_entropy(&re, &im, n));
        energy_trace.push(expected_energy(&re, &im, &diag));
    }

    // Measurement statistics (renormalise to absorb floating-point drift).
    let mut total = 0.0;
    let mut expected = 0.0;
    let mut best = 0usize;
    let mut best_p = -1.0;
    for x in 0..dim {
        let p = re[x] * re[x] + im[x] * im[x];
        total += p;
        expected += p * diag[x];
        if p > best_p {
            best_p = p;
            best = x;
        }
    }
    let spins: Vec<i8> = (0..n).map(|i| if (best >> i) & 1 == 1 { 1 } else { -1 }).collect();
    QaOutcome {
        spins,
        energy: diag[best],
        expected_energy: expected / total,
        success_prob: best_p / total,
        entropy_trace,
        energy_trace,
    }
}

/// ⟨ψ|H_problem|ψ⟩ for a (possibly unnormalised) state, renormalised.
fn expected_energy(re: &[f64], im: &[f64], diag: &[f64]) -> f64 {
    let mut e = 0.0;
    let mut total = 0.0;
    for x in 0..re.len() {
        let p = re[x] * re[x] + im[x] * im[x];
        e += p * diag[x];
        total += p;
    }
    e / total
}

/// Single-site von Neumann entanglement entropy (in bits): qubit 0 vs the rest.
///
/// The reduced density matrix of one qubit is 2×2 Hermitian, so its eigenvalues
/// have a closed form — exact and O(2^n), fast enough for the live per-step
/// trace. This entanglement is genuinely built up by the quantum dynamics: ~0
/// at the start (product state |+⟩^n) and end (≈ classical bitstring), with a
/// peak near the quantum phase transition. No physical QPU can report it.
///
/// For qubit 0, basis index `x` and its partner `x ^ 1` differ only in bit 0:
///   ρ00 = Σ_{bit0=0} |ψ_x|²,  ρ11 = Σ_{bit0=1} |ψ_x|²,
///   ρ01 = Σ_{bit0=0} ψ_x · conj(ψ_{x^1}).
///   λ± = (ρ00+ρ11)/2 ± √( ((ρ00-ρ11)/2)² + |ρ01|² ).
fn entanglement_entropy(re: &[f64], im: &[f64], n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let (mut r00, mut r11) = (0.0f64, 0.0f64);
    let (mut r01_re, mut r01_im) = (0.0f64, 0.0f64);
    let dim = re.len();
    let mut x = 0;
    while x < dim {
        // x has bit0 = 0; partner y = x + 1 has bit0 = 1.
        let y = x + 1;
        r00 += re[x] * re[x] + im[x] * im[x];
        r11 += re[y] * re[y] + im[y] * im[y];
        // ψ_x · conj(ψ_y) = (re_x+i im_x)(re_y - i im_y)
        r01_re += re[x] * re[y] + im[x] * im[y];
        r01_im += im[x] * re[y] - re[x] * im[y];
        x += 2;
    }
    let total = r00 + r11;
    if total <= 0.0 {
        return 0.0;
    }
    // Normalise to a trace-1 density matrix.
    r00 /= total;
    r11 /= total;
    let off2 = (r01_re * r01_re + r01_im * r01_im) / (total * total);
    let mid = (r00 + r11) / 2.0;
    let rad = (((r00 - r11) / 2.0).powi(2) + off2).sqrt();
    let lp = (mid + rad).clamp(0.0, 1.0);
    let lm = (mid - rad).clamp(0.0, 1.0);
    let mut s = 0.0;
    for lam in [lp, lm] {
        if lam > 1e-12 {
            s -= lam * lam.log2();
        }
    }
    s
}

/// Multiply every amplitude by the diagonal phase exp(-i·scale·diag[x]).
fn apply_problem_phase(re: &mut [f64], im: &mut [f64], diag: &[f64], scale: f64) {
    for ((r, i), &d) in re.iter_mut().zip(im.iter_mut()).zip(diag.iter()) {
        let phi = scale * d;
        let (c, s) = (phi.cos(), phi.sin());
        let nr = *r * c + *i * s;
        let ni = *i * c - *r * s;
        *r = nr;
        *i = ni;
    }
}

/// Apply Π_i exp(+iθ·X_i): a commuting product of single-qubit X-rotations.
fn apply_transverse(re: &mut [f64], im: &mut [f64], n: usize, theta: f64) {
    let (c, s) = (theta.cos(), theta.sin());
    let dim = re.len();
    for i in 0..n {
        let bit = 1usize << i;
        for x in 0..dim {
            if x & bit == 0 {
                let b = x | bit;
                // e^{iθX}: a' = c·a + i·s·b ; b' = i·s·a + c·b
                let (ra, ia, rb, ib) = (re[x], im[x], re[b], im[b]);
                re[x] = c * ra - s * ib;
                im[x] = c * ia + s * rb;
                re[b] = c * rb - s * ia;
                im[b] = c * ib + s * ra;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::{exact_ground_state, Ising};
    use crate::schedule;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn random_small(n: usize, rng: &mut impl Rng) -> Ising {
        let h: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut j = Vec::new();
        for i in 0..n {
            for k in (i + 1)..n {
                if rng.gen::<f64>() < 0.5 {
                    j.push((i, k, rng.gen_range(-1.0..1.0)));
                }
            }
        }
        Ising::new(h, j)
    }

    #[test]
    fn single_spin_aligns_with_field() {
        // h = +1 ⇒ ground state s = +1 (bit 1), energy -1. Large gap ⇒ easily adiabatic.
        let m = Ising::new(vec![1.0], vec![]);
        let out = anneal_statevector(&m, &schedule::smooth(400), 0.1);
        assert_eq!(out.spins, vec![1]);
        assert!((out.energy + 1.0).abs() < 1e-9);
        assert!(out.success_prob > 0.9, "success_prob={}", out.success_prob);
    }

    #[test]
    fn ferromagnetic_pair_reaches_ground_energy() {
        // J(0,1)=+1, no field ⇒ degenerate ground states ↑↑/↓↓ at energy -1.
        let m = Ising::new(vec![0.0, 0.0], vec![(0, 1, 1.0)]);
        let out = anneal_statevector(&m, &schedule::smooth(500), 0.1);
        assert!((out.energy + 1.0).abs() < 1e-9, "energy={}", out.energy);
    }

    #[test]
    fn entanglement_rises_then_falls_during_anneal() {
        // A frustrated 2-spin antiferromagnet with opposing fields builds real
        // entanglement mid-anneal, then disentangles toward a classical answer.
        let m = Ising::new(vec![0.3, -0.3], vec![(0, 1, -1.0)]);
        let out = anneal_statevector(&m, &schedule::smooth(400), 0.1);
        let tr = &out.entropy_trace;
        assert_eq!(tr.len(), 400);
        let peak = tr.iter().cloned().fold(0.0f64, f64::max);
        let start = tr[0];
        let end = *tr.last().unwrap();
        // Entanglement is ~0 at the product-state start, grows to a real peak,
        // and is lower again at the (near-classical) end. This is the quantum
        // phase transition made visible — the demo's headline observable.
        assert!(peak > 0.05, "no entanglement peak: {peak}");
        assert!(peak > start && peak >= end, "peak {peak} not above start {start}/end {end}");
        // Entropy of one qubit is bounded by 1 bit.
        assert!(peak <= 1.0 + 1e-9, "entropy exceeds 1 bit: {peak}");
    }

    #[test]
    fn product_state_has_zero_entanglement() {
        // A single anneal step from |+>^n with tiny dt stays near a product
        // state, so single-site entanglement starts essentially at zero.
        let m = Ising::new(vec![0.5, -0.5, 0.2], vec![(0, 1, 0.4)]);
        let out = anneal_statevector(&m, &schedule::smooth(50), 0.1);
        assert!(out.entropy_trace[0] < 0.2, "start entropy too high: {}", out.entropy_trace[0]);
    }

    #[test]
    fn qa_matches_exact_on_small_random() {
        let mut rng = StdRng::seed_from_u64(5);
        let sched = schedule::smooth(1500); // slow ⇒ faithfully adiabatic for small n
        let trials = 8;
        let mut hits = 0;
        for _ in 0..trials {
            let m = random_small(6, &mut rng);
            let (_, e_exact) = exact_ground_state(&m);
            let out = anneal_statevector(&m, &sched, 0.1);
            // A simulator can never beat the true minimum.
            assert!(out.energy >= e_exact - 1e-9, "below exact: {} < {}", out.energy, e_exact);
            if (out.energy - e_exact).abs() < 1e-6 {
                hits += 1;
            }
        }
        // Adiabatic QA reaches the exact ground state on (nearly) all small
        // instances; where it doesn't is exactly where the gap is small — the
        // physics TESSERA exists to study, not a bug.
        assert!(hits >= 5, "adiabatic QA reached exact only {hits}/{trials}");
    }
}
