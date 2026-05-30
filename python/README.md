# python/ — neural guidance (GNN) + benchmarks

> Phase 2. Python's turf: the AI layer + experiment harness.

## the GNN (`guide.py`, planned)

A **graph neural network** (PyTorch + PyTorch-Geometric) that reads the QUBO/Ising
graph (node features `h_i`, edge features `J_ij`) and predicts:

1. an **annealing schedule** `A(s), B(s)` tailored to the instance, and
2. a **warm-start** product state (initial spin biases) to seed the MPS.

Trained **amortized** over a distribution of instances (learn once → guide many),
either unsupervised with the Ising energy as the loss (physics-informed, PI-GNN
style) or supervised against exact/SA solutions on small instances.

The GNN's outputs feed the Rust schedule driver, which calls the CUDA tensor-network
engine. The GNN never touches the hot loop — it only *guides* it.

## benchmark harness (`bench.py`, planned)

Honest, reproducible comparison on identical instances:

| solver | what |
|--------|------|
| `exact`        | brute-force ground state (small n) — truth |
| `tessera-sa`   | classical simulated annealing (Rust baseline) |
| `tessera-qa`   | real quantum annealing (CUDA tensor-network engine) |
| `tessera-qa+gnn` | quantum annealing with GNN-guided schedule/warm-start |

Reports: solution quality vs exact, time-to-solution, MPS bond dimension `chi`,
truncation error, and the entanglement-entropy trace along the anneal.

## deps

See `requirements.txt`. The Rust core is built with `cargo`; the CUDA engine with `nvcc`/CMake.
