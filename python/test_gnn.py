#!/usr/bin/env python3
"""
Headless GNN smoke test (Phase 2). Verifies the physics-informed GNN actually
LEARNS: after training unsupervised on a small distribution, its predicted
spins beat a random baseline on unseen instances, and the relaxed energy loss
decreases. Fast (CPU, tiny). Run: python test_gnn.py
"""
import sys
import random
import torch

sys.path.insert(0, ".")
from gnn_guide import (IsingGNN, random_instance, train, predict_spins,
                       exact_ground_energy, relaxed_energy)


def main():
    random.seed(1); torch.manual_seed(1)
    n = 10
    train_set = [random_instance(n, 0.5, random.Random(i)) for i in range(80)]
    model = IsingGNN(hidden=24, layers=3)

    with torch.no_grad():
        before = sum(float(relaxed_energy(ins, model(ins))) for ins in train_set) / len(train_set)
    train(model, train_set, epochs=40, lr=5e-3, verbose=False)
    with torch.no_grad():
        after = sum(float(relaxed_energy(ins, model(ins))) for ins in train_set) / len(train_set)
    print(f"relaxed energy: before={before:+.3f}  after={after:+.3f}")

    rng = random.Random(999)
    gnn_better = 0; trials = 30; gnn_gap = 0.0; rand_gap = 0.0
    for _ in range(trials):
        ins = random_instance(n, 0.5, rng)
        e_exact = exact_ground_energy(ins)
        e_gnn = ins.energy(predict_spins(model, ins))
        e_rand = min(ins.energy(torch.where(torch.rand(n) > 0.5, 1.0, -1.0)) for _ in range(10))
        gnn_gap += (e_gnn - e_exact) / (abs(e_exact) + 1e-9)
        rand_gap += (e_rand - e_exact) / (abs(e_exact) + 1e-9)
        if e_gnn <= e_rand:
            gnn_better += 1
    gnn_gap /= trials; rand_gap /= trials
    print(f"mean gap to exact: GNN={gnn_gap:.3f}  random(best-of-10)={rand_gap:.3f}")
    print(f"GNN <= random on {gnn_better}/{trials} unseen instances")

    ok = (after < before - 0.1) and (gnn_gap < rand_gap) and (gnn_better >= trials * 0.6)
    print("GNN test:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
