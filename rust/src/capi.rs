//! C ABI for the Python vision/ML pipeline (`python/tessera_cam.py`) to drive
//! the Rust solvers via ctypes. This is what fuses the webcam demo into TESSERA:
//! Python builds the Ising graph from a camera frame and calls one of these to
//! solve it, switching classical <-> quantum live on the same frame.
//!
//! Solvers exposed:
//! - `tessera_solve_sa`: classical simulated annealing (best-of-restarts).
//! - `tessera_solve_qa`: real quantum annealing — exact state-vector oracle
//!   (small n only; the honest "real quantum" path).
//!
//! All arrays are caller-owned, contiguous, and outlive the call. Spins are
//! written as i8 in {-1, +1}.

use crate::ising::{sa_solve, simulated_annealing_from, Ising};
use crate::quantum::anneal_statevector;
use crate::schedule;
use std::slice;

/// Build an `Ising` from raw C arrays (shared by both entry points).
///
/// # Safety
/// `h` valid for `n` f64; `ei`,`ej`,`ew` valid for `m` each.
unsafe fn ising_from_raw(
    h: *const f64,
    n: usize,
    ei: *const usize,
    ej: *const usize,
    ew: *const f64,
    m: usize,
) -> Ising {
    let h = slice::from_raw_parts(h, n).to_vec();
    let j = if m > 0 && !ei.is_null() {
        let ei = slice::from_raw_parts(ei, m);
        let ej = slice::from_raw_parts(ej, m);
        let ew = slice::from_raw_parts(ew, m);
        (0..m).map(|k| (ei[k], ej[k], ew[k])).collect()
    } else {
        Vec::new()
    };
    Ising::new(h, j)
}

/// Classical simulated annealing. Returns the final Ising energy; writes ±1
/// spins into `out_s` (length `n`).
///
/// If `s0` is non-null (length `n`), it is used as a **warm-start** — one anneal
/// run is seeded from it (temporal coherence across webcam frames, far fewer
/// sweeps needed) and compared against `restarts` random runs; the best wins.
/// Pass NULL for the cold best-of-restarts behaviour.
///
/// # Safety
/// See `ising_from_raw`; `out_s` writable for `n` i8; `s0` null or readable for `n`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tessera_solve_sa(
    h: *const f64,
    n: usize,
    ei: *const usize,
    ej: *const usize,
    ew: *const f64,
    m: usize,
    restarts: usize,
    sweeps: usize,
    s0: *const i8,
    out_s: *mut i8,
) -> f64 {
    if h.is_null() || out_s.is_null() {
        return 0.0;
    }
    let model = ising_from_raw(h, n, ei, ej, ew, m);
    let sweeps = sweeps.max(1);
    let mut rng = rand::thread_rng();

    let (mut best_s, mut best_e) = sa_solve(&model, restarts.max(1), sweeps, 5.0, 0.01, &mut rng);
    if !s0.is_null() {
        // Warm-start run: low starting temperature (we expect a good seed).
        let seed = slice::from_raw_parts(s0, n).to_vec();
        let (ws, we) = simulated_annealing_from(&model, &seed, sweeps, 1.0, 0.01, &mut rng);
        if we < best_e {
            best_e = we;
            best_s = ws;
        }
    }
    slice::from_raw_parts_mut(out_s, n).copy_from_slice(&best_s);
    best_e
}

/// Real quantum annealing via the exact state-vector oracle. Returns the Ising
/// energy of the most-probable measurement; writes ±1 spins into `out_s` and the
/// annealing success probability into `*out_prob` (if non-null).
///
/// Optionally writes the per-step **entanglement-entropy trace** into
/// `out_entropy` and the **energy-expectation trace** into `out_energy` (each of
/// length `steps`, or NULL to skip). The entropy trace is the demo's headline:
/// real quantum entanglement rising then falling across the anneal, peaking at
/// the quantum phase transition — something no physical QPU can report.
///
/// NOTE: exact 2^n state vector — keep `n` small (<= ~24). The camera demo coarsens
/// the frame to this many regions for the quantum path. Larger n -> use SA, or the
/// CUDA MPS engine (Phase 1b).
///
/// # Safety
/// See `ising_from_raw`; `out_s` writable for `n` i8; `out_prob` null or writable;
/// `out_entropy`/`out_energy` null or writable for `steps` f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tessera_solve_qa(
    h: *const f64,
    n: usize,
    ei: *const usize,
    ej: *const usize,
    ew: *const f64,
    m: usize,
    steps: usize,
    dt: f64,
    out_s: *mut i8,
    out_prob: *mut f64,
    out_entropy: *mut f64,
    out_energy: *mut f64,
) -> f64 {
    if h.is_null() || out_s.is_null() {
        return 0.0;
    }
    if n > 24 {
        // Refuse silently-wrong huge allocations; caller must coarsen for QA.
        return f64::NAN;
    }
    let steps = steps.max(2);
    let model = ising_from_raw(h, n, ei, ej, ew, m);
    let sched = schedule::smooth(steps);
    let out = anneal_statevector(&model, &sched, dt);
    slice::from_raw_parts_mut(out_s, n).copy_from_slice(&out.spins);
    if !out_prob.is_null() {
        *out_prob = out.success_prob;
    }
    if !out_entropy.is_null() {
        slice::from_raw_parts_mut(out_entropy, steps).copy_from_slice(&out.entropy_trace);
    }
    if !out_energy.is_null() {
        slice::from_raw_parts_mut(out_energy, steps).copy_from_slice(&out.energy_trace);
    }
    out.energy
}

#[cfg(test)]
mod tests {
    use super::*;

    // A ferromagnetic pair with no field: ground energy -1, both solvers agree.
    unsafe fn call(qa: bool) -> (f64, [i8; 2]) {
        let h = [0.0f64, 0.0];
        let (ei, ej, ew) = ([0usize], [1usize], [1.0f64]);
        let mut s = [0i8; 2];
        let e = if qa {
            let mut p = 0.0;
            tessera_solve_qa(h.as_ptr(), 2, ei.as_ptr(), ej.as_ptr(), ew.as_ptr(), 1, 400, 0.1, s.as_mut_ptr(), &mut p, std::ptr::null_mut(), std::ptr::null_mut())
        } else {
            tessera_solve_sa(h.as_ptr(), 2, ei.as_ptr(), ej.as_ptr(), ew.as_ptr(), 1, 20, 200, std::ptr::null(), s.as_mut_ptr())
        };
        (e, s)
    }

    #[test]
    fn capi_sa_reaches_ground_energy() {
        let (e, s) = unsafe { call(false) };
        assert!((e + 1.0).abs() < 1e-9, "energy={e}");
        assert_eq!(s[0], s[1], "ferromagnetic pair must align");
    }

    #[test]
    fn capi_qa_reaches_ground_energy() {
        let (e, s) = unsafe { call(true) };
        assert!((e + 1.0).abs() < 1e-9, "energy={e}");
        assert_eq!(s[0], s[1], "ferromagnetic pair must align");
    }

    #[test]
    fn capi_qa_refuses_too_large_n() {
        let h = vec![0.0f64; 30];
        let mut s = vec![0i8; 30];
        let e = unsafe {
            tessera_solve_qa(h.as_ptr(), 30, std::ptr::null(), std::ptr::null(), std::ptr::null(), 0, 100, 0.1, s.as_mut_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
        };
        assert!(e.is_nan(), "should refuse n>24 for exact state vector");
    }
}
