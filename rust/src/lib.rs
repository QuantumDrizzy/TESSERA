//! TESSERA core (Rust).
//!
//! Neural-guided real quantum annealing via tensor networks, with a live webcam
//! demo. Systems backbone: the QUBO/Ising problem model, QUBO⇄Ising conversion,
//! adiabatic schedule generation, an INDEPENDENT classical simulated-annealing
//! baseline + exact solver (ground truth), an EXACT state-vector quantum-anneal
//! oracle (real quantum dynamics for small n), safe FFI to the scalable C++/CUDA
//! tensor-network engine, and a C ABI (`capi`) the Python vision pipeline calls
//! to solve camera-frame Ising graphs live (classical ⇄ quantum).
//!
//! Two quantum backends, one physics:
//!   - `quantum`  exact 2^n state vector on CPU — the reference oracle.
//!   - `ffi`/cuda MPS + TDVP on GPU — scalable, validated against the oracle.
//!
//! The classical SA here exists ONLY as an honest benchmark baseline so quantum
//! annealing can be compared against classical annealing on identical instances.
//! It shares no code with SESHAT.

pub mod capi;
pub mod ffi;
pub mod ising;
pub mod quantum;
pub mod qubo;
pub mod schedule;
