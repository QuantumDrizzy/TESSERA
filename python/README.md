# python/ — webcam demo + (later) neural guidance

Python owns the vision/ML layer and calls the Rust solvers over a C ABI via
ctypes. Bare metal: no web, no localhost.

## `tessera_cam.py` — live webcam segmentation (Phase 3, the fused demo)

Each camera frame → superpixels = Ising spins; the foreground/background mask is
the **ground state** of the Ising energy, solved live by either solver:

- **CLASSICAL** simulated annealing — full resolution (~400 regions).
- **QUANTUM** annealing — exact state-vector adiabatic evolution (real quantum
  dynamics). Exact `2ⁿ`, so the frame is coarsened to ≤ ~22 regions for this path.

Keys: `m` toggle SA↔QA · `+/-` quantum region count · `q` quit.

```bash
cd ../rust && cargo build --release      # builds tessera.dll (x64 Native Tools prompt)
pip install -r requirements.txt
python tessera_cam.py
```

This is the honest core of the demo: switch to quantum mode and you are watching
real adiabatic quantum dynamics segment your webcam in real time — at the low
resolution the exact state vector allows. Full resolution is the job of the
scalable CUDA MPS engine (Phase 1b) and GNN guidance (Phase 2).

> **OpenCV note:** SLIC superpixels require `opencv-contrib-python` (the
> `cv2.ximgproc` module). Plain `opencv-python` will raise
> `ModuleNotFoundError: No module named 'cv2.ximgproc'` — uninstall it and
> install the contrib build (see `requirements.txt`).

## `test_ffi.py` — headless FFI smoke test (no camera, no OpenCV)

Validates the ctypes ↔ Rust round-trip for both solvers on a synthetic Ising
graph. Run after building the core:

```bash
python test_ffi.py        # expects: FFI round-trip: PASS
```

## `gnn_guide.py` — physics-informed GNN guidance (Phase 2)

A graph neural network (pure PyTorch message passing — no torch-geometric
dependency) that reads the Ising graph (node feature `h_i`, edge feature `J_ij`)
and predicts a per-spin warm-start `P(s_i = +1)`. Trained **unsupervised** with
the relaxed Ising energy itself as the loss (physics-informed GNN, Schuetz et
al. 2022) — no labels, no solver in the training loop. Amortised: train once on
a distribution, predict a warm-start for any new instance in one forward pass.

```bash
python test_gnn.py                 # verifies it learns + beats random (PASS)
python gnn_guide.py --n 12         # train (CUDA if available) + report gap-to-exact
```

Verified: relaxed energy +0.01 → -7.31 while training; on unseen instances 0.21
mean gap to optimum vs 0.53 for best-of-10 random, beating random 24/30. Feed
`predict_spins()` into the Rust SA / C++ MPS annealer as the seed to refine it.
