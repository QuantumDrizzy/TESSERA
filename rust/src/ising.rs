//! Ising/QUBO problem model + classical baselines (ground truth & benchmark).
//!
//!   E(s) = - Σ_<i,j> J_ij s_i s_j  -  Σ_i h_i s_i ,    s_i ∈ {-1, +1}
//!
//! `exact_ground_state` is the source of truth (small n). `simulated_annealing`
//! is the INDEPENDENT classical baseline the quantum annealer is measured against.

use rand::Rng;

/// An Ising problem: external fields `h` and sparse couplings `j` (upper triangular).
pub struct Ising {
    pub n: usize,
    pub h: Vec<f64>,
    pub j: Vec<(usize, usize, f64)>,
}

impl Ising {
    pub fn new(h: Vec<f64>, j: Vec<(usize, usize, f64)>) -> Self {
        Ising { n: h.len(), h, j }
    }

    /// Total Ising energy of a spin configuration.
    pub fn energy(&self, s: &[i8]) -> f64 {
        let mut e = 0.0;
        for &(i, k, w) in &self.j {
            e -= w * s[i] as f64 * s[k] as f64;
        }
        for (&hi, &si) in self.h.iter().zip(s.iter()) {
            e -= hi * si as f64;
        }
        e
    }

    /// Per-spin adjacency (for fast local-field updates in SA).
    fn adjacency(&self) -> Vec<Vec<(usize, f64)>> {
        let mut adj = vec![Vec::new(); self.n];
        for &(i, k, w) in &self.j {
            adj[i].push((k, w));
            adj[k].push((i, w));
        }
        adj
    }
}

/// Exact ground state by brute force — the source of truth. Small `n` only.
pub fn exact_ground_state(model: &Ising) -> (Vec<i8>, f64) {
    assert!(model.n <= 24, "brute force is only for n <= 24");
    let mut best = vec![1i8; model.n];
    let mut best_e = f64::INFINITY;
    for bits in 0u64..(1u64 << model.n) {
        let s: Vec<i8> = (0..model.n)
            .map(|b| if (bits >> b) & 1 == 1 { 1 } else { -1 })
            .collect();
        let e = model.energy(&s);
        if e < best_e {
            best_e = e;
            best = s;
        }
    }
    (best, best_e)
}

/// One simulated-annealing run from a given start state (geometric schedule).
fn sa_run(
    model: &Ising,
    mut s: Vec<i8>,
    sweeps: usize,
    t0: f64,
    t_end: f64,
    rng: &mut impl Rng,
) -> (Vec<i8>, f64) {
    let adj = model.adjacency();
    for sweep in 0..sweeps {
        let frac = sweep as f64 / (sweeps.max(2) - 1) as f64;
        let t = t0 * (t_end / t0).powf(frac);
        for i in 0..model.n {
            let mut field = model.h[i];
            for &(k, w) in &adj[i] {
                field += w * s[k] as f64;
            }
            let de = 2.0 * s[i] as f64 * field; // ΔE if spin i flips
            if de < 0.0 || rng.gen::<f64>() < (-de / t).exp() {
                s[i] = -s[i];
            }
        }
    }
    let e = model.energy(&s);
    (s, e)
}

/// Classical thermal simulated annealing (Metropolis) from a random start —
/// independent baseline. Geometric temperature schedule from `t0` to `t_end`.
pub fn simulated_annealing(
    model: &Ising,
    sweeps: usize,
    t0: f64,
    t_end: f64,
    rng: &mut impl Rng,
) -> (Vec<i8>, f64) {
    let s = (0..model.n).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect();
    sa_run(model, s, sweeps, t0, t_end, rng)
}

/// Simulated annealing from a given warm-start state (temporal coherence).
pub fn simulated_annealing_from(
    model: &Ising,
    s0: &[i8],
    sweeps: usize,
    t0: f64,
    t_end: f64,
    rng: &mut impl Rng,
) -> (Vec<i8>, f64) {
    sa_run(model, s0.to_vec(), sweeps, t0, t_end, rng)
}

/// Best-of-`restarts` simulated annealing — the classical baseline *solver*
/// (the benchmark harness uses this, not a single SA run).
pub fn sa_solve(
    model: &Ising,
    restarts: usize,
    sweeps: usize,
    t0: f64,
    t_end: f64,
    rng: &mut impl Rng,
) -> (Vec<i8>, f64) {
    let mut best_s = vec![1i8; model.n];
    let mut best_e = f64::INFINITY;
    for _ in 0..restarts.max(1) {
        let (s, e) = simulated_annealing(model, sweeps, t0, t_end, rng);
        if e < best_e {
            best_e = e;
            best_s = s;
        }
    }
    (best_s, best_e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn random_instance(n: usize, p: f64, rng: &mut impl Rng) -> Ising {
        let h: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut j = Vec::new();
        for i in 0..n {
            for k in (i + 1)..n {
                if rng.gen::<f64>() < p {
                    j.push((i, k, rng.gen_range(-1.0..1.0)));
                }
            }
        }
        Ising::new(h, j)
    }

    #[test]
    fn sa_solver_reaches_exact_ground_state_on_small_instances() {
        let mut rng = StdRng::seed_from_u64(1);
        let trials = 20;
        let mut hits = 0;
        for _ in 0..trials {
            let m = random_instance(12, 0.5, &mut rng);
            let (_, e_exact) = exact_ground_state(&m);
            let (_, e_sa) = sa_solve(&m, 40, 400, 5.0, 0.01, &mut rng);
            if (e_sa - e_exact).abs() < 1e-6 {
                hits += 1;
            }
        }
        // Best-of-restarts SA must be a solid baseline: exact on (nearly) all
        // small frustrated instances. Where it *can't* is exactly where quantum
        // annealing earns its keep — the comparison TESSERA is built to measure.
        assert!(hits >= 19, "SA-solver reached exact only {hits}/{trials} times");
    }

    #[test]
    fn sa_beats_random_configuration() {
        let mut rng = StdRng::seed_from_u64(2);
        let m = random_instance(16, 0.5, &mut rng);
        let rand_s: Vec<i8> = (0..m.n).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect();
        let (_, e_sa) = simulated_annealing(&m, 400, 3.0, 0.01, &mut rng);
        assert!(e_sa < m.energy(&rand_s), "SA did not beat a random configuration");
    }
}
