//! Safe FFI to the C++/CUDA tensor-network quantum-annealing engine (`cuda/`).
//!
//! Matches the C ABI declared in `cuda/README.md`. Feature-gated on `cuda`:
//! - default build (no `cuda` feature): `quantum_anneal` returns
//!   `Err(QaError::EngineNotLinked)`, so the crate builds & tests without a GPU;
//! - `--features cuda`: the extern symbol is linked (see `build.rs`) and called.

use crate::ising::Ising;
use crate::schedule::Schedule;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QaError {
    /// The CUDA engine was not compiled/linked (build with `--features cuda`).
    EngineNotLinked,
    /// The schedule had fewer than 2 steps.
    BadSchedule,
}

#[cfg(feature = "cuda")]
extern "C" {
    fn tessera_qa_anneal(
        h: *const f64,
        n: usize,
        ei: *const usize,
        ej: *const usize,
        ew: *const f64,
        m: usize,
        schedule_a: *const f64,
        schedule_b: *const f64,
        steps: usize,
        chi: usize,
        out_s: *mut i8,
        out_entropy: *mut f64,
    ) -> f64;
}

/// Result of a quantum anneal: final spins, final energy, and the entanglement-
/// entropy trace along the schedule (the "watch the quantum happen" signal).
pub struct QaResult {
    pub spins: Vec<i8>,
    pub energy: f64,
    pub entropy: Vec<f64>,
}

/// Run real quantum annealing on `model` using `sched`, with MPS bond dimension
/// `chi` (the exactness/cost knob — always report it alongside results).
pub fn quantum_anneal(model: &Ising, sched: &Schedule, chi: usize) -> Result<QaResult, QaError> {
    let steps = sched.steps();
    if steps < 2 {
        return Err(QaError::BadSchedule);
    }

    #[cfg(feature = "cuda")]
    {
        let ei: Vec<usize> = model.j.iter().map(|t| t.0).collect();
        let ej: Vec<usize> = model.j.iter().map(|t| t.1).collect();
        let ew: Vec<f64> = model.j.iter().map(|t| t.2).collect();
        let mut spins = vec![0i8; model.n];
        let mut entropy = vec![0.0f64; steps];
        let energy = unsafe {
            tessera_qa_anneal(
                model.h.as_ptr(),
                model.n,
                ei.as_ptr(),
                ej.as_ptr(),
                ew.as_ptr(),
                model.j.len(),
                sched.a.as_ptr(),
                sched.b.as_ptr(),
                steps,
                chi,
                spins.as_mut_ptr(),
                entropy.as_mut_ptr(),
            )
        };
        Ok(QaResult { spins, energy, entropy })
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Keep the signature honest and the args "used" without a GPU present.
        let _ = (model, chi);
        Err(QaError::EngineNotLinked)
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;
    use crate::schedule;

    #[test]
    fn quantum_anneal_reports_not_linked_without_cuda_feature() {
        let m = Ising::new(vec![1.0, -1.0], vec![(0, 1, 0.5)]);
        let sched = schedule::smooth(16);
        assert_eq!(quantum_anneal(&m, &sched, 32).err(), Some(QaError::EngineNotLinked));
    }

    #[test]
    fn quantum_anneal_rejects_degenerate_schedule() {
        let m = Ising::new(vec![1.0], vec![]);
        let bad = Schedule { a: vec![1.0], b: vec![0.0] };
        assert_eq!(quantum_anneal(&m, &bad, 32).err(), Some(QaError::BadSchedule));
    }
}
