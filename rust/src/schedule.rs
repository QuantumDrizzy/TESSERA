//! Adiabatic annealing schedules.
//!
//! The transverse-field Ising Hamiltonian is
//!   H(s) = -A(s)·Σ σ^x  -  B(s)·H_problem,   s : 0 → 1
//! starting in the transverse field (A≫B, easy ground state) and ending in the
//! problem Hamiltonian (B≫A). A schedule is the discretised pair A(s), B(s).

/// Discretised schedule: `a[k]` = transverse weight, `b[k]` = problem weight.
pub struct Schedule {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

impl Schedule {
    pub fn steps(&self) -> usize {
        self.a.len()
    }
}

/// Linear interpolation: A(s) = 1 - s, B(s) = s.
pub fn linear(steps: usize) -> Schedule {
    let n = steps.max(2);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for k in 0..n {
        let s = k as f64 / (n - 1) as f64;
        a.push(1.0 - s);
        b.push(s);
    }
    Schedule { a, b }
}

/// Smooth schedule: A(s) = cos²(πs/2), B(s) = sin²(πs/2). Gentler near the
/// endpoints (smaller dH/ds where the gap is usually smallest), and A+B ≡ 1.
pub fn smooth(steps: usize) -> Schedule {
    let n = steps.max(2);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for k in 0..n {
        let s = k as f64 / (n - 1) as f64;
        let theta = std::f64::consts::FRAC_PI_2 * s;
        a.push(theta.cos().powi(2));
        b.push(theta.sin().powi(2));
    }
    Schedule { a, b }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_endpoints_and_monotonicity(sch: &Schedule) {
        let n = sch.steps();
        assert!((sch.a[0] - 1.0).abs() < 1e-12 && sch.b[0].abs() < 1e-12);
        assert!(sch.a[n - 1].abs() < 1e-12 && (sch.b[n - 1] - 1.0).abs() < 1e-12);
        for k in 1..n {
            assert!(sch.a[k] <= sch.a[k - 1] + 1e-12, "A must be non-increasing");
            assert!(sch.b[k] >= sch.b[k - 1] - 1e-12, "B must be non-decreasing");
        }
    }

    #[test]
    fn linear_schedule_is_valid() {
        check_endpoints_and_monotonicity(&linear(64));
    }

    #[test]
    fn smooth_schedule_is_valid_and_normalised() {
        let sch = smooth(64);
        check_endpoints_and_monotonicity(&sch);
        for k in 0..sch.steps() {
            assert!((sch.a[k] + sch.b[k] - 1.0).abs() < 1e-12, "A+B must equal 1");
        }
    }
}
