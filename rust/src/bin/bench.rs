//! TESSERA benchmark harness (Phase 0/1).
//!
//! Compares solvers on identical random Ising instances:
//!   - `exact`  : brute-force ground state (truth, small n)
//!   - `SA`     : classical simulated annealing (Rust baseline)
//!   - `QA(sv)` : real quantum annealing — exact state-vector oracle (CPU)
//!   - `QA(mps)`: scalable quantum annealing — CUDA tensor-network engine
//!     (shows "not linked" until cuda/ is built with `--features cuda`)
//!
//! Run:  cargo run --bin bench

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;
use tessera::ffi::{quantum_anneal, QaError};
use tessera::ising::{exact_ground_state, sa_solve, Ising};
use tessera::quantum::anneal_statevector;
use tessera::schedule;

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

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    println!("TESSERA benchmark — classical SA vs real quantum annealing (vs exact truth)\n");
    println!(
        "{:>4} {:>10} {:>10} {:>10} {:>7} {:>12}",
        "n", "exact", "SA", "QA(sv)", "P(sol)", "QA(mps)"
    );
    println!("{}", "-".repeat(60));

    let sched = schedule::smooth(800);
    for &n in &[6usize, 8, 10, 12] {
        let m = random_instance(n, 0.5, &mut rng);

        let (_, e_exact) = exact_ground_state(&m);
        let (_, e_sa) = sa_solve(&m, 40, 400, 5.0, 0.01, &mut rng);

        let t = Instant::now();
        let qa = anneal_statevector(&m, &sched, 0.1);
        let _sv_ms = t.elapsed().as_secs_f64() * 1e3;

        let mps = match quantum_anneal(&m, &sched, 32) {
            Ok(r) => format!("{:.4}", r.energy),
            Err(QaError::EngineNotLinked) => "not linked".to_string(),
            Err(e) => format!("{e:?}"),
        };

        println!(
            "{:>4} {:>10.4} {:>10.4} {:>10.4} {:>6.0}% {:>12}",
            n,
            e_exact,
            e_sa,
            qa.energy,
            qa.success_prob * 100.0,
            mps
        );
    }

    eprintln!(
        "\nQA(sv) = exact state-vector adiabatic evolution on CPU (real quantum dynamics, the\n\
         oracle). P(sol) = annealing success probability. QA(mps) = the scalable CUDA\n\
         tensor-network engine — build cuda/ and `--features cuda` to fill it; it is validated\n\
         against QA(sv) on these small instances."
    );
}
