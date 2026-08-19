# -*- coding: utf-8 -*-
"""Gera examples/demo_word_embeddings.ipynb."""
import json
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = [
md("""# GSVD alignment angle on word embeddings (PT-BR vs English)

The same **gsvdlib** pipeline, no images anywhere: **A** holds MUSE-aligned
fastText embeddings (300d) of Portuguese words and **B** the embeddings of
their English translations. Because the MUSE vectors live in one shared
multilingual space, the GSVD frame tells us which directions of that space
are *shared* between the two languages and which are *language-specific* —
and theta(z) scores any embedding by how PT-like or EN-like it is.

Data: top-20k vectors of `wiki.multi.pt.vec` / `wiki.multi.en.vec` plus the
MUSE en-pt dictionary, cached by `_fetch_muse.py` (run it once first)."""),

code("""from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from gsvdlib import (ArrayDataset, prepare_data, theta_angles,
                     metrics_from_angles, linear_cka,
                     get_angles_C, highlight_closest, get_most_similar_row_H)
from gsvdlib.plots import LineRenderer, plot_samples, plot_angles, \\
    plot_angle_histogram, plot_posterior

data = np.load(Path.home() / ".gsvdlib" / "muse" / "muse_pt_en_top20000.npz")
pt_words, pt_X = list(data["pt_words"]), data["pt_X"].astype(float)
en_words, en_X = list(data["en_words"]), data["en_X"].astype(float)
print("PT:", pt_X.shape, " EN:", en_X.shape)"""),

md("""## 1. Paired vocabulary → an `ArrayDataset`

We keep translation pairs where both sides are in the top-20k vocabularies,
are alphabetic and distinct, then split words into train/test. The generic
`ArrayDataset` takes it from there — the rest of the notebook is exactly the
same code path used for MNIST and CIFAR-10."""),

code("""pt_idx = {w: i for i, w in enumerate(pt_words)}
en_idx = {w: i for i, w in enumerate(en_words)}

pairs, seen_en, seen_pt = [], set(), set()
for en_w, pt_w in zip(data["dict_en"], data["dict_pt"]):
    en_w, pt_w = str(en_w), str(pt_w)
    if (en_w in en_idx and pt_w in pt_idx
            and en_w.isalpha() and pt_w.isalpha()
            and len(en_w) > 2 and len(pt_w) > 2
            and en_w != pt_w                      # drop identical cognates
            and en_w not in seen_en and pt_w not in seen_pt):
        pairs.append((en_w, pt_w))
        seen_en.add(en_w)
        seen_pt.add(pt_w)
print(len(pairs), "usable en-pt pairs")
print("examples:", pairs[100:106])

rng = np.random.default_rng(1234)
order = rng.permutation(len(pairs))
n_train = 1200
train_p, test_p = order[:n_train], order[n_train:n_train + 1000]

def block(idxs):
    en_sel = [pairs[i][0] for i in idxs]
    pt_sel = [pairs[i][1] for i in idxs]
    X = np.hstack([pt_X[:, [pt_idx[w] for w in pt_sel]],
                   en_X[:, [en_idx[w] for w in en_sel]]])
    labels = np.array(["pt"] * len(pt_sel) + ["en"] * len(en_sel))
    words = pt_sel + en_sel
    return X, labels, words

X_tr, y_tr, words_tr = block(train_p)
X_te, y_te, words_te = block(test_p)

ds = ArrayDataset(splits={"train": (X_tr, y_tr), "test": (X_te, y_te)},
                  names={"pt": "Portuguese", "en": "English"})"""),

code("""# "samples" are now word vectors: draw a few with the LineRenderer
show = [0, 1, 2, 3, 4]
fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 2.2))
line = LineRenderer(lw=0.8)
for ax, j in zip(axes, show):
    line(ax, X_tr[:, j], title=f"'{words_tr[j]}' (pt)")
plt.tight_layout()"""),

md("""## 2. GSVD of A (Portuguese) vs B (English)"""),

code("""prep = prepare_data(ds, "pt", "en", n_A=900, n_B=800, seed=1234)
g = prep.gsvd

print("U:", g.U.shape, " V_til:", g.V_til.shape, " H:", g.H.shape)
print(f"blocks: bl={g.bl}  br={g.br}  wl={g.wl}  wr={g.wr}  r={g.r}")
errA = np.linalg.norm(prep.A.T - g.U @ g.C @ g.H)
errB = np.linalg.norm(prep.B.T - g.V_til @ g.S @ g.H)
print(f"||A' - U C H|| = {errA:.2e}   ||B' - V_til S H|| = {errB:.2e}")
print(f"linear CKA(pt, en) = {linear_cka(prep.A, prep.B):.4f}")"""),

code("""plot_angles(g.C, target_deg=45);"""),

md("""## 3. Interpreting the shared directions with words

Each shared row of `H` is a direction of the multilingual space. To read it,
we list the vocabulary words whose embeddings are most aligned with it
(`get_most_similar_row_H` against the full 20k PT and EN vocabularies) — the
embedding analogue of "plotting the H image" in the MNIST notebook."""),

code("""angles = get_angles_C(g.C)
psc = int(highlight_closest(angles, 45, 1)[0])
picks = [g.bl, g.bl + 1, psc, g.bl + g.r - 2, g.bl + g.r - 1]

for i in picks:
    h = g.H[i, :]
    sim_pt = get_most_similar_row_H(h, pt_X.T)
    sim_en = get_most_similar_row_H(h, en_X.T)
    top_pt = [pt_words[j] for j in np.argsort(-np.abs(sim_pt))[:6]]
    top_en = [en_words[j] for j in np.argsort(-np.abs(sim_en))[:6]]
    print(f"H[{i}]  theta = {angles[i]:5.1f}\\u00b0")
    print("   pt:", ", ".join(top_pt))
    print("   en:", ", ".join(top_en))"""),

md("""## 4. Classifying held-out words by theta(z)

For each unseen embedding: is it better explained by the Portuguese frame or
the English one?"""),

code("""# theta for every held-out word, keeping the angle <-> word correspondence
Xte_pt = X_te[:, y_te == "pt"]
Xte_en = X_te[:, y_te == "en"]
words_te_pt = [w for w, y in zip(words_te, y_te) if y == "pt"]
words_te_en = [w for w, y in zip(words_te, y_te) if y == "en"]

angles_pt = theta_angles(Xte_pt, g.C, g.S, g.H)
angles_en = theta_angles(Xte_en, g.C, g.S, g.H)
print(f"PT: mean theta = {angles_pt.mean():.2f}\\u00b0 (\\u03c3={angles_pt.std():.2f})   "
      f"EN: mean theta = {angles_en.mean():.2f}\\u00b0 (\\u03c3={angles_en.std():.2f})")"""),

code("""plot_angle_histogram(angles_pt, angles_en, "Portuguese", "English");
plot_posterior(angles_pt, angles_en, "Portuguese", "English");"""),

code("""m = metrics_from_angles(angles_pt, angles_en)
print(f"accuracy = {m['accuracy']:.4f}")
for side, name in (("A", "Portuguese"), ("B", "English")):
    print(f"{name:10s}  precision={m[f'precision_{side}']:.4f}"
          f"  recall={m[f'recall_{side}']:.4f}  f1={m[f'f1_{side}']:.4f}")"""),

code("""# the most PT-like and most EN-like test words according to theta
i_pt = np.argsort(angles_pt)
i_en = np.argsort(angles_en)[::-1]
print("PT words with lowest theta (most Portuguese-like):")
print("  ", ", ".join(words_te_pt[j] for j in i_pt[:10]))
print("EN words with highest theta (most English-like):")
print("  ", ", ".join(words_te_en[j] for j in i_en[:10]))
print("PT words with highest theta (look 'English' to the frame):")
print("  ", ", ".join(words_te_pt[j] for j in i_pt[::-1][:10]))
print("EN words with lowest theta (look 'Portuguese' to the frame):")
print("  ", ", ".join(words_te_en[j] for j in i_en[::-1][:10]))"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i}"

path = Path(__file__).parent / "demo_word_embeddings.ipynb"
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)

# valida sintaxe de todas as celulas de codigo antes de executar
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
print("written + syntax ok:", path)
