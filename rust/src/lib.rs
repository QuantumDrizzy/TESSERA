//! TESSERA core (Rust).
//!
//! Systems backbone: the QUBO/Ising problem model, QUBO⇄Ising conversion,
//! adiabatic schedule generation, an INDEPENDENT classical simulated-annealing
//! baseline + exact solver (ground truth), an EXACT state-vector quantum-anneal
//! oracle (real quantum dynamics for small n), and safe FFI to the scalable
//! C++/CUDA tensor-network quantum-annealing engine.
//!
//! Two quantum backends, one physics:
//!   - `quantum`  exact 2^n state vector on CPU — the reference oracle.
//!   - `ffi`/cuda MPS + TDVP on GPU — scalable, validated against the oracle.
//!
//! The classical SA here exists ONLY as an honest benchmark baseline so quantum
//! annealing can be compared against classical annealing on identical instances.
//! It shares no code with SESHAT.

pub mod ffi;
pub mod ising;
pub mod quantum;
pub mod qubo;
pub mod schedule;
