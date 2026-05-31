#!/usr/bin/env python3
"""
TESSERA Phase 2 — GNN guidance for Ising/QUBO annealing.

The AI pillar of the flagship. A graph neural network reads the Ising graph
(node feature h_i, edge feature J_ij) and predicts a per-spin probability
p_i = P(s_i = +1). The relaxed energy

    E_relax(p) = - sum_<i,j> J_ij (2p_i-1)(2p_j-1) - sum_i h_i (2p_i-1)

is differentiable, so the GNN trains UNSUPERVISED with the Ising energy itself
as the loss (physics-informed GNN, Schuetz et al., Nat. Mach. Intell. 2022).
No labels, no solver in the loop.

One Ising graph, three views — this is the third:
    GNN learns it  ·  tensor network represents its quantum state  ·  annealing
    searches its ground state.

Honest framing of the value:
  - AMORTISATION: train once over a distribution of instances, then predict a
    warm-start for any new instance in a single forward pass (milliseconds).
  - WARM-START: the GNN's rounded prediction seeds the annealer, which then
    refines it. A GNN is not magic; sometimes greedy ties it. The win is the
    amortised warm-start, measured, not claimed.

Pure PyTorch message passing (no torch-geometric dependency). Right tool per
domain: Python/torch for learning, Rust+C++ for the hot annealing loop.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Ising:
    n: int
    h: torch.Tensor
    ei: torch.Tensor
    ej: torch.Tensor
    ew: torch.Tensor

    def energy(self, spins: torch.Tensor) -> float:
        e = -(self.ew * spins[self.ei] * spins[self.ej]).sum() - (self.h * spins).sum()
        return float(e)


def random_instance(n: int, p_edge: float, rng: random.Random, device="cpu") -> Ising:
    h = torch.tensor([rng.uniform(-1, 1) for _ in range(n)], dtype=torch.float32, device=device)
    ei, ej, ew = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                ei.append(i); ej.append(j); ew.append(rng.uniform(-1, 1))
    if not ei:
        ei, ej, ew = [0], [min(1, n - 1)], [rng.uniform(-1, 1)]
    return Ising(
        n=n, h=h,
        ei=torch.tensor(ei, dtype=torch.long, device=device),
        ej=torch.tensor(ej, dtype=torch.long, device=device),
        ew=torch.tensor(ew, dtype=torch.float32, device=device),
    )


def exact_ground_energy(ins: Ising) -> float:
    n = ins.n
    best = math.inf
    h = ins.h.cpu(); ei = ins.ei.cpu(); ej = ins.ej.cpu(); ew = ins.ew.cpu()
    for x in range(1 << n):
        s = torch.tensor([1.0 if (x >> b) & 1 else -1.0 for b in range(n)])
        e = float(-(ew * s[ei] * s[ej]).sum() - (h * s).sum())
        best = min(best, e)
    return best


class IsingGNN(nn.Module):
    """Edge-conditioned message passing -> per-node P(s_i = +1)."""

    def __init__(self, hidden: int = 32, layers: int = 3):
        super().__init__()
        self.embed = nn.Linear(1, hidden)
        self.msg = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        )
        self.upd = nn.ModuleList(nn.GRUCell(hidden, hidden) for _ in range(layers))
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ins: Ising) -> torch.Tensor:
        x = self.embed(ins.h.unsqueeze(-1))
        src = torch.cat([ins.ei, ins.ej])
        dst = torch.cat([ins.ej, ins.ei])
        ew = torch.cat([ins.ew, ins.ew]).unsqueeze(-1)
        for msg, upd in zip(self.msg, self.upd):
            m = msg(torch.cat([x[src], ew], dim=-1))
            agg = torch.zeros_like(x).index_add_(0, dst, m)
            x = upd(agg, x)
        return torch.sigmoid(self.readout(x)).squeeze(-1)


def relaxed_energy(ins: Ising, p: torch.Tensor) -> torch.Tensor:
    s = 2.0 * p - 1.0
    quad = (ins.ew * s[ins.ei] * s[ins.ej]).sum()
    lin = (ins.h * s).sum()
    return -(quad + lin)


def train(model: IsingGNN, instances: list[Ising], epochs: int, lr: float, verbose=True) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        random.shuffle(instances)
        total = 0.0
        for ins in instances:
            opt.zero_grad()
            p = model(ins)
            loss = relaxed_energy(ins, p)
            ent = -(p * torch.log(p + 1e-9) + (1 - p) * torch.log(1 - p + 1e-9)).mean()
            (loss + 0.5 * ent).backward()
            opt.step()
            total += float(loss.detach())
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  mean relaxed energy {total/len(instances):+.4f}")


@torch.no_grad()
def predict_spins(model: IsingGNN, ins: Ising) -> torch.Tensor:
    model.eval()
    p = model(ins)
    return torch.where(p >= 0.5, 1.0, -1.0)


def main():
    ap = argparse.ArgumentParser(description="TESSERA GNN guidance (Phase 2)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-size", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--p-edge", type=float, default=0.5)
    ap.add_argument("--eval-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dev = args.device
    print(f"TESSERA GNN guidance — device={dev}, n={args.n}")

    train_set = [random_instance(args.n, args.p_edge, rng, dev) for _ in range(args.train_size)]
    model = IsingGNN().to(dev)
    print(f"Training on {len(train_set)} instances ({args.epochs} epochs)...")
    train(model, train_set, args.epochs, args.lr)

    print(f"\nEvaluating on {args.eval_size} unseen instances (n={args.n}):")
    gnn_gaps, rand_gaps, exact_hits, better = [], [], 0, 0
    for _ in range(args.eval_size):
        ins = random_instance(args.n, args.p_edge, rng, dev)
        e_exact = exact_ground_energy(ins) if args.n <= 18 else None
        e_gnn = ins.energy(predict_spins(model, ins))
        e_rand = min(ins.energy(torch.where(torch.rand(ins.n, device=dev) > 0.5, 1.0, -1.0))
                     for _ in range(10))
        if e_exact is not None:
            d = abs(e_exact) + 1e-9
            gnn_gaps.append((e_gnn - e_exact) / d)
            rand_gaps.append((e_rand - e_exact) / d)
            if abs(e_gnn - e_exact) < 1e-6:
                exact_hits += 1
            if e_gnn <= e_rand:
                better += 1

    if gnn_gaps:
        print(f"  GNN   mean gap to exact : {sum(gnn_gaps)/len(gnn_gaps):.3f}")
        print(f"  rand  mean gap to exact : {sum(rand_gaps)/len(rand_gaps):.3f}  (best-of-10)")
        print(f"  GNN hit exact ground    : {exact_hits}/{args.eval_size}")
        print(f"  GNN <= random           : {better}/{len(gnn_gaps)}")
        print("\nHonest read: the GNN gives a learned warm-start in one forward pass")
        print("(amortised over the trained distribution); feed predict_spins() into")
        print("the Rust SA / C++ MPS annealer as the seed to refine it.")


if __name__ == "__main__":
    main()
