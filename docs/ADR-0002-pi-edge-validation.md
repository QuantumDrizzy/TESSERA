# ADR-0002: Edge / Raspberry Pi validation of the webcam Ising-segmentation demo

**Status:** Proposed
**Date:** 2026-06-19
**Deciders:** Antonio (QuantumDrizzy)

## Context

The hardware era (see GROUNDWATCH `ADR-0001` / `project-groundwatch`) makes a personal
Raspberry Pi 5 the bench for validating existing projects on real hardware. TESSERA is a
strong candidate to go *first*: it already has the fused webcam demo (`python/tessera_cam.py`,
Phase 3, tested) — live image segmentation as the **ground state of an Ising model**, solved by
classical SA **or** real (state-vector) quantum annealing, with a live entanglement-entropy
panel. It's pinned and public. Each frame → SLIC superpixels → N Ising spins → solve → fg/bg
mask → composite. The Rust `tessera` cdylib does the solving; Python (OpenCV) owns the vision.

Goal: **no new code** — port, harden, and validate the existing demo on the Pi with a real
webcam, and measure honest numbers. This is the first "software → hardware, validated" milestone.

## Decision

Run TESSERA's webcam demo on the Pi as the first hardware validation, with two clearly-separated
paths:
- **SA (classical) = the real-time edge path** — `N_SP_CLASSICAL = 400` superpixels, warm-started
  frame to frame. This is what actually runs live on the Pi.
- **QA (state-vector) = the small-n physics demo** — exact 2ⁿ on CPU, capped (`QUANTUM_CAP = 22`),
  showing the entanglement-entropy spike at the phase transition. A *visualisation*, not a fast solver.

The **CUDA MPS-TEBD engine (`cuda/`) does NOT run on the Pi** (no NVIDIA GPU) — and it isn't needed
for the demo. The Pi validates the SA edge path + the small-n QA panel; the scalable MPS stays a
desktop/Jetson concern.

## What to touch (profiling — when the Pi lands)

1. **Rust cdylib → aarch64.** `tessera` is pure Rust for the SA + CPU state-vector solvers (no GPU
   dependency on that path) → cross-compile / build natively for `aarch64-unknown-linux-gnu`.
   `tessera_cam.py` already locates `libtessera.so` in `rust/target/{release,debug}` — just build it on the Pi.
2. **OpenCV on ARM.** The demo needs `opencv-contrib-python` (`cv2.ximgproc.SLIC` + Farneback flow).
   Verify the contrib wheel installs on the Pi's Python; if SLICO is painful on ARM, fall back to a
   lighter superpixel (felzenszwalb / a grid) — the Ising pipeline is agnostic to the segmenter.
3. **Camera.** USB webcam at index 0 works as-is (`cv2.VideoCapture(0)`); a Pi CSI camera needs the
   libcamera/V4L2 shim. Either is a one-line change.
4. **Measure honestly (the deliverable):** SA segmentation **FPS at N=400 on the Pi** (the real edge
   number), and the **max QA n** tractable in the Pi's RAM/CPU. Report; do not assert.

## Honest limits `[KNOWN_LIMIT]`

- QA mode is exact 2ⁿ on CPU → tiny n on a Pi; it's the *physics demo*, not a solver.
- No CUDA on the Pi → the MPS engine and its `--features cuda` path are out of scope here.
- "Real quantum annealing" applies only to the small-n state-vector mode; the live edge path is
  classical SA. Keep the on-screen tags (`CLASSICAL` / `QUANTUM`) honest, as they already are.

## Consequences
- **Easier:** a real, demoable edge result on day one (webcam → Ising segmentation, live) that
  documents as a build-in-public Twitter thread (the KHAOS arc): unbox → first frame → SA live →
  flip to QA + the entanglement panel.
- **Shared substrate:** the Ising/QUBO core is the same thesis as DRIFT, SESHAT and temp0r — one
  Hamiltonian, many faces. TESSERA is its vision face.

## Action Items
1. [ ] Build the `tessera` cdylib on the Pi (aarch64); confirm `tessera_cam.py` loads it.
2. [ ] Get `opencv-contrib-python` (or a lighter superpixel fallback) working on the Pi.
3. [ ] Run live with a USB/CSI camera; measure SA FPS @ N=400 and max QA n.
4. [ ] Record the build-in-public thread.
