"""Choose the truncation of the pseudo-inverse in theta(z) = f(H^+ z) on validation data.

theta_angles solves H^T c = z by least squares. H is ill-conditioned (pixels that
almost never carry ink in training give tiny singular values), so ink falling in
those pixels blows c up and theta stops reflecting the shape of the sample.
Truncating the pseudo-inverse (singular values below rcond * s_max set to zero)
removes those directions.

Protocol (pooled centering, the gsvdlib default):
  * 8 class pairs (the 4 MNIST pairs of the paper + the 4 Fashion-MNIST pairs of
    the site) x 5 draws of the 900/800 training base;
  * validation set = the next 1000 training images of each class in the same
    random permutation, so it is disjoint from the base and from the test split;
  * rcond is chosen by the mean validation AUC over all pairs and draws;
  * only then the balanced test set (seed 4321) compares the chosen rcond with
    the untruncated pseudo-inverse: AUC, accuracy at 45 degrees and accuracy at
    the threshold calibrated on validation.

Writes examples/truncation_validation.json.
"""
import json
import os

import numpy as np

from gsvdlib import FashionMNISTDataset, MNISTDataset, prepare_data
from gsvdlib.blocks import to_intersection
from gsvdlib.datasets import balanced_count, sample_pair

PAIRS = {"mnist": [(1, 5), (0, 7), (4, 9), (3, 9)],
         "fashion": [(0, 4), (2, 3), (7, 9), (0, 7)]}
BASE_SEEDS = (1234, 1, 2, 3, 4)
N_A, N_B, N_VAL = 900, 800, 1000
RCONDS = (0.0, 1e-4, 1e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def auc(t_a, t_b):
    """P(theta_B > theta_A), ties counted half."""
    return float((t_b[:, None] > t_a[None]).mean() + 0.5 * (t_b[:, None] == t_a[None]).mean())


def balanced_acc(t_a, t_b, thr):
    return float(((t_a < thr).mean() + (t_b >= thr).mean()) / 2)


def calibrate(t_a, t_b):
    grid = np.round(np.arange(0.0, 90.05, 0.1), 1)
    acc = ((t_a[None] < grid[:, None]).mean(1) + (t_b[None] >= grid[:, None]).mean(1)) / 2
    best = np.flatnonzero(acc == acc.max())
    return float(grid[best[len(best) // 2]])


def validation_split(ds, a, b, seed):
    rng = np.random.default_rng(seed)  # replays sample_pair's draw, then takes the next N_VAL
    out = []
    for label, n in ((a, N_A), (b, N_B)):
        X = ds.get_class(label, split="train")
        out.append(X[:, rng.permutation(X.shape[1])[n:n + N_VAL]])
    return out


def main():
    datasets = {"mnist": MNISTDataset(), "fashion": FashionMNISTDataset()}
    runs = []
    for fam, pairs in PAIRS.items():
        ds = datasets[fam]
        for a, b in pairs:
            n = balanced_count(ds, a, b, split="test")
            TA, TB = sample_pair(ds, a, b, n, n, split="test", seed=4321)
            for seed in BASE_SEEDS:
                prep = prepare_data(ds, a, b, n_A=N_A, n_B=N_B, seed=seed)
                g, mu = prep.gsvd, prep.center_test[:, None]
                VA, VB = validation_split(ds, a, b, seed)
                U, s, Vt = np.linalg.svd(g.H.T, full_matrices=False)
                Ci, Si = to_intersection(g.C, g.S)

                def theta(X, rcond):
                    keep = s > rcond * s[0] if rcond > 0 else s > 0
                    c = Vt[keep].T @ ((U[:, keep].T @ (X - mu)) / s[keep, None])
                    return np.degrees(np.arctan2(np.linalg.norm(Si @ c, axis=0),
                                                 np.linalg.norm(Ci @ c, axis=0)))

                row = {"family": fam, "pair": f"{a}v{b}", "seed": seed, "by_rcond": {}}
                for r in RCONDS:
                    va, vb, ta, tb = theta(VA, r), theta(VB, r), theta(TA, r), theta(TB, r)
                    tau = calibrate(va, vb)
                    row["by_rcond"][str(r)] = {
                        "kept": int((s > r * s[0]).sum() if r > 0 else (s > 0).sum()),
                        "val_auc": auc(va, vb), "test_auc": auc(ta, tb),
                        "test_acc45": balanced_acc(ta, tb, 45.0), "tau": tau,
                        "test_acc_tau": balanced_acc(ta, tb, tau),
                    }
                runs.append(row)
                print(f"{fam} {a}v{b} seed {seed}: val AUC " + " ".join(
                    f"{row['by_rcond'][str(r)]['val_auc']:.3f}" for r in RCONDS), flush=True)

    def mean(key, r, fam=None):
        v = [x["by_rcond"][str(r)][key] for x in runs if fam in (None, x["family"])]
        return float(np.mean(v))

    print("\nrcond      val AUC (MNIST / Fashion / all)   directions kept (MNIST mean)")
    for r in RCONDS:
        print(f"{r:<8g}   {mean('val_auc', r, 'mnist'):.4f} / {mean('val_auc', r, 'fashion'):.4f}"
              f" / {mean('val_auc', r):.4f}        {mean('kept', r, 'mnist'):.0f}")
    best = max(RCONDS, key=lambda r: mean("val_auc", r))
    print(f"\nchosen on validation: rcond = {best:g}")

    print("\nTEST, mean ± sd over the 5 draws  (untruncated  →  chosen rcond)")
    summary = {}
    for fam, pairs in PAIRS.items():
        for a, b in pairs:
            sel = [x for x in runs if x["family"] == fam and x["pair"] == f"{a}v{b}"]
            line = f"{fam:7s} {a}v{b} "
            summary[f"{fam}_{a}v{b}"] = {}
            for key in ("test_auc", "test_acc45", "test_acc_tau"):
                v0 = [x["by_rcond"]["0.0"][key] for x in sel]
                v1 = [x["by_rcond"][str(best)][key] for x in sel]
                summary[f"{fam}_{a}v{b}"][key] = [float(np.mean(v0)), float(np.mean(v1))]
                line += f"| {key[5:]:7s} {np.mean(v0):.4f}±{np.std(v0):.4f} → {np.mean(v1):.4f}±{np.std(v1):.4f} "
            print(line)
    wins = sum(x["by_rcond"][str(best)]["test_auc"] > x["by_rcond"]["0.0"]["test_auc"] for x in runs)
    print(f"\ntest AUC improves in {wins}/{len(runs)} pair x draw combinations")

    out = os.path.join(os.path.dirname(__file__), "truncation_validation.json")
    with open(out, "w") as f:
        json.dump({"rconds": RCONDS, "chosen": best, "summary": summary, "runs": runs}, f)


if __name__ == "__main__":
    main()
