#!/usr/bin/env python3
"""
Headless FFI smoke test — validates the ctypes <-> Rust round-trip WITHOUT a
camera or OpenCV. Checks that both solvers reach the ground state through the C
ABI, and that the quantum path returns a real entanglement-entropy trace that
rises then falls (the quantum phase transition). Run:  python test_ffi.py
"""
import os
import sys
import ctypes
import numpy as np

def load():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for sub in ("release", "debug"):
        for name in ("tessera.dll", "libtessera.so", "libtessera.dylib"):
            p = os.path.join(root, "rust", "target", sub, name)
            if os.path.exists(p):
                return ctypes.CDLL(p)
    sys.exit("tessera core not built. Run:  cd ../rust && cargo build --release")

lib = load()
f64 = ctypes.POINTER(ctypes.c_double); usz = ctypes.POINTER(ctypes.c_size_t); i8 = ctypes.POINTER(ctypes.c_int8)
lib.tessera_solve_sa.restype = ctypes.c_double
lib.tessera_solve_sa.argtypes = [f64, ctypes.c_size_t, usz, usz, f64, ctypes.c_size_t,
                                 ctypes.c_size_t, ctypes.c_size_t, i8, i8]  # ...restarts,sweeps,s0,out_s
lib.tessera_solve_qa.restype = ctypes.c_double
lib.tessera_solve_qa.argtypes = [f64, ctypes.c_size_t, usz, usz, f64, ctypes.c_size_t,
                                 ctypes.c_size_t, ctypes.c_double, i8, f64, f64, f64]  # ...steps,dt,out_s,prob,entropy,energy

def p(a, t): return a.ctypes.data_as(ctypes.POINTER(t))

# Ferromagnetic chain of 6 spins, no field: ground state all-aligned, E = -(n-1).
n = 6
h = np.zeros(n, np.float64)
ei = np.arange(0, n - 1, dtype=np.uintp)
ej = np.arange(1, n, dtype=np.uintp)
ew = np.ones(n - 1, np.float64)
m = len(ei)
expected = -float(n - 1)

# Classical (cold: s0 = NULL)
out = np.empty(n, np.int8)
e_sa = lib.tessera_solve_sa(p(h, ctypes.c_double), n, p(ei, ctypes.c_size_t), p(ej, ctypes.c_size_t),
                            p(ew, ctypes.c_double), m, 30, 300, None, p(out, ctypes.c_int8))
sa_aligned = len(set(out.tolist())) == 1
print(f"SA : E={e_sa:.4f} (expect {expected})  spins={out.tolist()}  aligned={sa_aligned}")

# Classical warm-start: feed the previous solution back in (few sweeps).
e_sa2 = lib.tessera_solve_sa(p(h, ctypes.c_double), n, p(ei, ctypes.c_size_t), p(ej, ctypes.c_size_t),
                             p(ew, ctypes.c_double), m, 4, 60, p(out, ctypes.c_int8), p(out, ctypes.c_int8))
print(f"SA warm-start: E={e_sa2:.4f} (expect {expected})  [few sweeps, seeded]")

# Quantum (state vector) with entanglement + energy traces.
steps = 600
out2 = np.empty(n, np.int8); prob = ctypes.c_double(0.0)
ent = np.empty(steps, np.float64); en = np.empty(steps, np.float64)
e_qa = lib.tessera_solve_qa(p(h, ctypes.c_double), n, p(ei, ctypes.c_size_t), p(ej, ctypes.c_size_t),
                            p(ew, ctypes.c_double), m, steps, 0.1, p(out2, ctypes.c_int8),
                            ctypes.byref(prob), p(ent, ctypes.c_double), p(en, ctypes.c_double))
qa_aligned = len(set(out2.tolist())) == 1
peak = float(ent.max())
print(f"QA : E={e_qa:.4f} (expect {expected})  P(sol)={prob.value*100:.0f}%  aligned={qa_aligned}")
print(f"     entanglement: start={ent[0]:.3f}  peak={peak:.3f}  end={ent[-1]:.3f}  (bits, 1 qubit)")
print(f"     energy trace: {en[0]:.2f} -> {en[-1]:.2f}")

ok = (abs(e_sa - expected) < 1e-9 and sa_aligned and
      abs(e_sa2 - expected) < 1e-9 and
      abs(e_qa - expected) < 1e-9 and qa_aligned and
      peak > 0.02 and peak >= ent[0] and peak >= ent[-1])
print("FFI round-trip:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
