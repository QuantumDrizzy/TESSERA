//! QUBO ⇄ Ising conversion.
//!
//! QUBO minimises  f(x) = Σ_i a_i x_i + Σ_{i<j} b_ij x_i x_j ,  x_i ∈ {0, 1}.
//! Substituting x_i = (1 + s_i)/2,  s_i ∈ {-1, +1}, gives an Ising model whose
//! ground state is the QUBO minimiser. We track the constant `offset` so the
//! true objective value can be recovered:  f(x) = E_ising(s) + offset.

use crate::ising::Ising;

/// A QUBO problem (upper-triangular quadratic form).
pub struct Qubo {
    pub n: usize,
    pub linear: Vec<f64>,                 // a_i  (diagonal Q_ii)
    pub quad: Vec<(usize, usize, f64)>,   // (i, j, b_ij) with i < j
}

impl Qubo {
    pub fn new(linear: Vec<f64>, quad: Vec<(usize, usize, f64)>) -> Self {
        Qubo { n: linear.len(), linear, quad }
    }

    /// Objective value for a binary assignment.
    pub fn evaluate(&self, x: &[u8]) -> f64 {
        let mut f = 0.0;
        for (&a, &xi) in self.linear.iter().zip(x.iter()) {
            f += a * xi as f64;
        }
        for &(i, j, b) in &self.quad {
            f += b * x[i] as f64 * x[j] as f64;
        }
        f
    }

    /// Convert to an Ising model. Returns `(ising, offset)` with
    /// `f(x) = ising.energy(s) + offset` for `s = 2x - 1`.
    pub fn to_ising(&self) -> (Ising, f64) {
        let n = self.n;
        let mut h = vec![0.0f64; n];
        let mut offset = 0.0f64;

        for (hi, &a) in h.iter_mut().zip(self.linear.iter()) {
            *hi += -a / 2.0;
            offset += a / 2.0;
        }
        let mut j = Vec::with_capacity(self.quad.len());
        for &(a, b, bij) in &self.quad {
            j.push((a, b, -bij / 4.0));
            h[a] += -bij / 4.0;
            h[b] += -bij / 4.0;
            offset += bij / 4.0;
        }
        (Ising::new(h, j), offset)
    }
}

/// Ising spins (±1) → QUBO bits (0/1).
pub fn spins_to_bits(s: &[i8]) -> Vec<u8> {
    s.iter().map(|&x| if x > 0 { 1 } else { 0 }).collect()
}

/// QUBO bits (0/1) → Ising spins (±1).
pub fn bits_to_spins(x: &[u8]) -> Vec<i8> {
    x.iter().map(|&b| if b == 1 { 1 } else { -1 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn qubo_to_ising_preserves_objective() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..50 {
            let n = 8;
            let linear: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
            let mut quad = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng.gen::<f64>() < 0.5 {
                        quad.push((i, j, rng.gen_range(-2.0..2.0)));
                    }
                }
            }
            let q = Qubo::new(linear, quad);
            let (ising, offset) = q.to_ising();

            // Random binary assignment: f(x) must equal E_ising(s) + offset.
            let x: Vec<u8> = (0..n).map(|_| rng.gen::<bool>() as u8).collect();
            let s = bits_to_spins(&x);
            let lhs = q.evaluate(&x);
            let rhs = ising.energy(&s) + offset;
            assert!((lhs - rhs).abs() < 1e-9, "mismatch: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn qubo_minimiser_matches_ising_ground_state() {
        use crate::ising::exact_ground_state;
        let mut rng = StdRng::seed_from_u64(4);
        let n = 10;
        let linear: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut quad = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                quad.push((i, j, rng.gen_range(-1.0..1.0)));
            }
        }
        let q = Qubo::new(linear, quad);
        let (ising, offset) = q.to_ising();
        let (s_star, e_star) = exact_ground_state(&ising);

        // Brute-force the QUBO directly and compare the optima.
        let mut best_f = f64::INFINITY;
        for bits in 0u32..(1u32 << n) {
            let x: Vec<u8> = (0..n).map(|b| ((bits >> b) & 1) as u8).collect();
            best_f = best_f.min(q.evaluate(&x));
        }
        assert!((best_f - (e_star + offset)).abs() < 1e-9);
        // and the recovered assignment achieves it
        let x_star = spins_to_bits(&s_star);
        assert!((q.evaluate(&x_star) - best_f).abs() < 1e-9);
    }
}
