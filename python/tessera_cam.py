#!/usr/bin/env python3
"""
TESSERA — live webcam segmentation as the ground state of an Ising model,
solved by either classical simulated annealing or REAL quantum annealing.

The fused webcam demo: Python owns the vision pipeline (OpenCV capture, SLIC
superpixels, optical flow, compositing) and calls the Rust solvers (`tessera`
cdylib) over a C ABI via ctypes. Bare metal: no web, no localhost.

Each frame -> N superpixels = N Ising spins:
    E(s) = - sum_<i,j> J_ij s_i s_j  -  sum_i h_i s_i ,   s_i in {-1 bg, +1 fg}
  h_i  (data term)  : motion (optical flow) + center prior + border contrast
  J_ij (smoothness) : colour-similar neighbours couple

Solve the ground state -> fg/bg mask -> composite live.

QUANTUM mode additionally shows, in a live panel, the REAL entanglement entropy
built up across the anneal -- it rises then falls, peaking at the quantum phase
transition. No physical QPU can report this.

Keys:
  q : quit
  m : toggle solver  (SA classical  <->  QA quantum state-vector)
  +/- : (quantum mode) more/fewer regions  [capped so 2^n stays tractable]

Build first (x64 Native Tools prompt):  cd ../rust && cargo build --release
Run:                                     python tessera_cam.py
"""
import os
import sys
import ctypes
import numpy as np
import cv2

# -- load the Rust core (cdylib) --------------------------------------------
def _load_core():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for sub in ("release", "debug"):
        for name in ("tessera.dll", "libtessera.so", "libtessera.dylib"):
            p = os.path.join(root, "rust", "target", sub, name)
            if os.path.exists(p):
                return ctypes.CDLL(p)
    sys.exit("tessera core not built. Run:  cd ../rust && cargo build --release")

_LIB = _load_core()
_f64 = ctypes.POINTER(ctypes.c_double)
_usz = ctypes.POINTER(ctypes.c_size_t)
_i8 = ctypes.POINTER(ctypes.c_int8)

_LIB.tessera_solve_sa.restype = ctypes.c_double
_LIB.tessera_solve_sa.argtypes = [
    _f64, ctypes.c_size_t, _usz, _usz, _f64, ctypes.c_size_t,   # h,n, ei,ej,ew,m
    ctypes.c_size_t, ctypes.c_size_t, _i8, _i8,                  # restarts, sweeps, s0, out_s
]
_LIB.tessera_solve_qa.restype = ctypes.c_double
_LIB.tessera_solve_qa.argtypes = [
    _f64, ctypes.c_size_t, _usz, _usz, _f64, ctypes.c_size_t,   # h,n, ei,ej,ew,m
    ctypes.c_size_t, ctypes.c_double, _i8, _f64, _f64, _f64,    # steps, dt, out_s, prob, entropy, energy
]

def _p(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))

def _prep(h, ei, ej, ew):
    h = np.ascontiguousarray(h, np.float64)
    ei = np.ascontiguousarray(ei, np.uintp)
    ej = np.ascontiguousarray(ej, np.uintp)
    ew = np.ascontiguousarray(ew, np.float64)
    return h, ei, ej, ew

def solve_sa(h, ei, ej, ew, restarts=8, sweeps=120, s0=None):
    n, m = len(h), len(ei)
    h, ei, ej, ew = _prep(h, ei, ej, ew)
    out = np.empty(n, np.int8)
    s0p = _p(np.ascontiguousarray(s0, np.int8), ctypes.c_int8) if (s0 is not None and len(s0) == n) else None
    e = _LIB.tessera_solve_sa(_p(h, ctypes.c_double), n, _p(ei, ctypes.c_size_t),
                              _p(ej, ctypes.c_size_t), _p(ew, ctypes.c_double), m,
                              restarts, sweeps, s0p, _p(out, ctypes.c_int8))
    return out, float(e), None, None

def solve_qa(h, ei, ej, ew, steps=300, dt=0.1):
    n, m = len(h), len(ei)
    h, ei, ej, ew = _prep(h, ei, ej, ew)
    out = np.empty(n, np.int8); prob = ctypes.c_double(0.0)
    ent = np.empty(steps, np.float64); en = np.empty(steps, np.float64)
    e = _LIB.tessera_solve_qa(_p(h, ctypes.c_double), n, _p(ei, ctypes.c_size_t),
                              _p(ej, ctypes.c_size_t), _p(ew, ctypes.c_double), m,
                              steps, dt, _p(out, ctypes.c_int8), ctypes.byref(prob),
                              _p(ent, ctypes.c_double), _p(en, ctypes.c_double))
    return out, float(e), float(prob.value), (ent, en)

# -- vision pipeline (OpenCV / NumPy -- favourite turf) ---------------------
PROC_W = 480
N_SP_CLASSICAL = 400     # SA scales fine: full resolution
N_SP_QUANTUM = 16        # QA is exact 2^n -> must stay small
QUANTUM_CAP = 22
W_MOTION, W_CENTER, W_CONTRAST, W_SMOOTH, COLOR_SIGMA = 1.4, 0.5, 0.8, 1.2, 12.0

def superpixels(bgr, n_target):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    region = int(np.sqrt(bgr.shape[0] * bgr.shape[1] / max(n_target, 1)))
    slic = cv2.ximgproc.createSuperpixelSLIC(lab, cv2.ximgproc.SLICO, max(region, 8))
    slic.iterate(8)
    return slic.getLabels(), slic.getNumberOfSuperpixels()

def adjacency(labels):
    a = np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], 1)
    b = np.stack([labels[:-1, :].ravel(), labels[1:, :].ravel()], 1)
    e = np.concatenate([a, b], 0)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(np.sort(e, axis=1), axis=0)

def region_stats(labels, n, lab, fmag):
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0
    mean_lab = np.stack([np.bincount(flat, lab[..., c].ravel(), minlength=n) / counts for c in range(3)], 1)
    ys, xs = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
    cx = np.bincount(flat, xs.ravel(), minlength=n) / counts
    cy = np.bincount(flat, ys.ravel(), minlength=n) / counts
    motion = np.bincount(flat, fmag.ravel(), minlength=n) / counts
    return mean_lab, np.stack([cx, cy], 1), motion

def build_h(mean_lab, cent, motion, shape, border):
    H, W = shape
    mo = motion / (motion.max() + 1e-6)
    d = np.sqrt(((cent[:, 0] - W / 2) / (W / 2)) ** 2 + ((cent[:, 1] - H / 2) / (H / 2)) ** 2)
    center = 1 - np.clip(d, 0, 1)
    bmean = mean_lab[border].mean(0)
    contrast = np.linalg.norm(mean_lab - bmean, axis=1)
    contrast = contrast / (contrast.max() + 1e-6)
    bias = 0.5 * (W_MOTION + W_CENTER + W_CONTRAST)
    return (W_MOTION * mo + W_CENTER * center + W_CONTRAST * contrast - bias)

def smoothness(edges, mean_lab):
    cd = np.linalg.norm(mean_lab[edges[:, 0]] - mean_lab[edges[:, 1]], axis=1)
    w = W_SMOOTH * np.exp(-(cd ** 2) / (2 * COLOR_SIGMA ** 2))
    return edges[:, 0], edges[:, 1], w

def composite(frame, labels, s, tag, color):
    fg = (s > 0)
    raw = (fg[labels].astype(np.uint8) * 255)
    mask = cv2.GaussianBlur(raw, (7, 7), 0)
    mm = (mask.astype(np.float32) / 255)[..., None]
    blurred = cv2.GaussianBlur(frame, (0, 0), 8)
    out = (frame * mm + blurred * (1 - mm)).astype(np.uint8)
    edge = cv2.morphologyEx(raw, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    glow = cv2.GaussianBlur(cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR), (0, 0), 6)
    glow = (glow.astype(np.float32) * np.array(color)).clip(0, 255).astype(np.uint8)
    out = cv2.add(out, glow)
    cv2.putText(out, tag, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(c * 255 for c in color), 2)
    return out

def draw_quantum_panel(img, entropy, energy, prob):
    """Overlay a live panel: entanglement-entropy curve (the phase transition)
    plus P(sol) and the converging energy. entropy is in bits (0..1, 1 qubit)."""
    H, W = img.shape[:2]
    pw, ph = 260, 120
    x0, y0 = 12, H - ph - 12
    sub = (img[y0:y0 + ph, x0:x0 + pw].astype(np.float32) * 0.25).astype(np.uint8)
    img[y0:y0 + ph, x0:x0 + pw] = sub
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (90, 90, 90), 1)

    n = len(entropy)
    if n > 1:
        emax = max(float(entropy.max()), 1e-6)
        pts = []
        for i in range(n):
            px = x0 + 8 + int((pw - 16) * i / (n - 1))
            py = y0 + ph - 24 - int((ph - 40) * (entropy[i] / emax))
            pts.append((px, py))
        for i in range(1, n):
            cv2.line(img, pts[i - 1], pts[i], (255, 220, 60), 1, cv2.LINE_AA)
        pk = int(np.argmax(entropy))           # the quantum phase transition
        cv2.circle(img, pts[pk], 3, (60, 60, 255), -1)
    cv2.putText(img, f"entanglement (peak {float(entropy.max()):.2f} bit)", (x0 + 8, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 60), 1, cv2.LINE_AA)
    cv2.putText(img, f"P(sol) {prob*100:.0f}%   E {energy[-1]:.2f}", (x0 + 8, y0 + ph - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("No webcam at index 0")
    quantum = False
    n_q = N_SP_QUANTUM
    prev = None
    s_prev = None          # warm-start (classical mode)
    print("TESSERA webcam -- [m] SA<->QA   [+/-] quantum regions   [q] quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H0, W0 = frame.shape[:2]
        small = cv2.resize(frame, (PROC_W, int(H0 * PROC_W / W0)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is None:
            prev = gray
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
        fmag = np.linalg.norm(flow, axis=2)
        prev = gray

        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float64)
        n_target = n_q if quantum else N_SP_CLASSICAL
        labels, n = superpixels(small, n_target)
        edges = adjacency(labels)
        mean_lab, cent, motion = region_stats(labels, n, lab, fmag)
        border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        h = build_h(mean_lab, cent, motion, small.shape[:2], border)
        ei, ej, ew = smoothness(edges, mean_lab)

        if quantum:
            s, e, prob, traces = solve_qa(h, ei, ej, ew)
            tag = f"QUANTUM annealing (state-vector)  n={n}"
            color = (1.0, 0.45, 0.1)             # blue-ish (BGR scaled 0..1)
            s_prev = None                         # state vector restarts from |+>
        else:
            s, e, prob, traces = solve_sa(h, ei, ej, ew, s0=s_prev)
            tag = f"CLASSICAL simulated annealing  n={n}"
            color = (0.2, 1.0, 0.3)              # green-ish
            s_prev = s if len(s) == n else None   # warm-start next frame

        out = composite(small, labels, s, tag, color)
        out = cv2.resize(out, (W0, H0))
        if quantum and traces is not None:
            draw_quantum_panel(out, traces[0], traces[1], prob)
        cv2.imshow("TESSERA - Ising segmentation (q quits)", out)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            quantum = not quantum
            s_prev = None
        elif k in (ord('+'), ord('=')):
            n_q = min(n_q + 2, QUANTUM_CAP)
        elif k == ord('-'):
            n_q = max(n_q - 2, 4)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
