# gsvdlib

Python port of the Julia pipeline from **GSVD for Geometry-Grounded Dataset
Comparison: An Alignment Angle Is All You Need**
([arXiv:2603.10283](https://arxiv.org/abs/2603.10283), experiments repo:
`gsvd-alignment-angle`).

Naming follows the paper: the cosine factor is **C** (formerly `D1`) and the
sine factor is **S** (formerly `D2`).

## Install

```bash
pip install -e .          # from the repo root
pip install -e .[dev]     # + pytest
```

## The GSVD core

SciPy does not expose LAPACK's `ggsvd3` (what Julia's `svd(A, B)` calls), so
`gsvdlib.lapack_gsvd.gsvd_factors` reimplements the factorization with the
Paige–Saunders construction (pivoted QR of `[A; B]` + CS decomposition via
SVD of the top block), returning the same contract: `A = U C R0 Qᵀ`,
`B = V S R0 Qᵀ`, with LAPACK's block layout and ordering.

`gsvdlib.gsvd(A, B)` is the paper's `Our_SVD`: it applies the block
permutation that puts `S` in the blocked layout (`I(bl)` / middle `C̃, S̃` /
`I(br)`) and returns a `GSVDResult` dataclass instead of a 14-tuple:

```python
import numpy as np
from gsvdlib import gsvd

A = np.random.randn(100, 80)
B = np.random.randn(90, 80)

res = gsvd(A, B).sorted()      # .sorted() = sort_and_rebuild (angles 0° → 90°)
res.U @ res.C @ res.H          # == A
res.V_til @ res.S @ res.H      # == B
res.bl, res.br, res.wl, res.wr, res.r   # block sizes
```

Also available, mirroring the Julia functions: `make_C_S`, `make_A_B`,
`wire_size`, `permutation`, `sort_and_rebuild`, `set_to_0_closest_to_45`,
`infer_block_sizes`, `zero_pure_blocks`, `to_intersection`,
`get_nonzero_per_column`, `get_angles_C`, `get_angles_S`,
`highlight_closest`, `get_most_similar_row_H`.

## Generic datasets

Everything downstream works on "vectors as columns" through the
`VectorDataset` protocol — nothing is MNIST-specific:

```python
from gsvdlib import MNISTDataset, FashionMNISTDataset, ArrayDataset

ds = MNISTDataset()                      # downloads IDX files once
ds = FashionMNISTDataset()

# word embeddings, activations, any matrix + labels:
ds = ArrayDataset(X, labels, names={0: "animals", 1: "tools"})
```

## Pipeline

```python
from gsvdlib import MNISTDataset, prepare_data, evaluate_pair, run_pair_experiment

ds = MNISTDataset()
prep = prepare_data(ds, 4, 9, n_A=900, n_B=800)      # sample, center, gsvd().sorted()
angles_A, angles_B, acc = evaluate_pair(ds, 4, 9, prep)  # test samples centered too

rows, angle_data = run_pair_experiment(ds, [(1, 5), (0, 7), (4, 9), (3, 9)])
# pandas.DataFrame(rows) -> precision / recall / F1 / accuracy / linear CKA
```

θ(z) is computed by `theta_angles` / `classify` / `classify_set`
(vectorized least squares, restricted to col(A) ∩ col(B) by default).

**Centering.** By default (`centering="pooled"`) A, B and every new sample z
are centered by the same vector, the mean of A and B together
(`prep.center_test`). Subtract it from your own samples before calling
`theta_angles`:

```python
Z = ds.get_class(4, "test")[:, :5] - prep.center_test[:, None]
theta_angles(Z, prep.gsvd.C, prep.gsvd.S, prep.gsvd.H)
```

`centering="per_class"` reproduces the ICLR 2026 protocol (A and B each
centered by their own mean, new samples left uncentered). On the four MNIST
pairs the two protocols differ by less than one point of accuracy.

## Generic plots

Sample rendering is abstracted behind *renderers*, so the same plot functions
serve images, embeddings, or anything else:

```python
from gsvdlib.plots import (ImageRenderer, LineRenderer, plot_samples,
                           plot_recon, plot_angles, plot_angle_histogram,
                           plot_posterior, animate_components)

r28 = ImageRenderer((28, 28))
plot_samples(prep.A, r28, k=20, mean=prep.center_A)
plot_angle_histogram(angles_A, angles_B, "Digit 4", "Digit 9")
plot_posterior(angles_A, angles_B, "Digit 4", "Digit 9")
animate_components(prep.gsvd.H.T, r28, save="H.gif")
# embeddings? just swap the renderer:
plot_samples(emb_matrix, LineRenderer())
```

## Validation

`pytest` covers reconstruction, orthogonality, `CᵀC + SᵀS = I`, rank-deficient
pairs, block inference and the classification pipeline. Running the four
paper MNIST pairs reproduces the Julia metrics (`metrics_paper_pairs.csv`)
to within sampling noise (Julia and NumPy RNGs draw different subsets; CKA
matches to ~3 decimals).
