# -*- coding: utf-8 -*-
# %% [markdown]
# # Experimento 5 — Onde no mBERT o espaço é mais interlingual?
#
# Extraímos embeddings estáticos (palavra isolada, média dos subtokens) de
# **todas as 13 camadas** do `bert-base-multilingual-cased` para os mesmos
# pares PT/EN usados no frame MUSE de `paper_common`. Para cada camada L
# montamos um frame GSVD conjunto (`build_frame`) e medimos:
#
# - **CKA linear** entre as matrizes centradas A (PT) e B (EN);
# - o espectro de ângulos θ (`get_angles_C`): desvio-padrão (largura) e
#   fração de direções na **banda compartilhada** |θ−45°| ≤ 5°;
# - **P@1 de retrieval de tradução** (400 pares de teste, cosseno no espaço
#   completo da camada, centrado com as médias da base).
#
# Comparamos com o baseline MUSE (frame estático de `paper_common`).
# A literatura sugere que as camadas intermediárias (~8) são as mais
# "interlinguais".

# %%
import json
from pathlib import Path

import numpy as np

from gsvdlib import get_angles_C, linear_cka
from paper_common import Frame, build_frame, RESULTS

MBERT_CACHE = Path.home() / ".gsvdlib" / "mbert"
MBERT_CACHE.mkdir(parents=True, exist_ok=True)
CACHE_NPZ = MBERT_CACHE / "mbert_pairs_layers.npz"

N_BASE, N_TEST = 1200, 400
SEED = 1234

# %% [markdown]
# ## Palavras pareadas
#
# Reconstruímos a mesma permutação (semente 1234) usada dentro de `Frame`
# para recuperar os 1200 pares da base; os 400 de teste vêm de
# `f.test_pairs`. Cada par é (en, pt).

# %%
f = Frame()
order = np.random.default_rng(SEED).permutation(len(f.pairs))
base_pairs = [f.pairs[i] for i in order[:N_BASE]]
assert f.test_pairs[0] == f.pairs[order[N_BASE]]  # sanidade: mesma ordem
test_pairs = f.test_pairs[:N_TEST]

pairs_all = base_pairs + test_pairs
en_words = [e for e, p in pairs_all]
pt_words = [p for e, p in pairs_all]

# %% [markdown]
# ## Extração mBERT (com cache)
#
# Cada palavra é tokenizada isolada (sem contexto); para cada camada 0..12
# tiramos a média dos hidden states dos subtokens reais (sem [CLS]/[SEP]).
# Resultado: arrays (13, 768, 1600) por língua, salvos em um único npz.

# %%
def extract_mbert(words, tokenizer, model, batch_size=64):
    import torch
    n = len(words)
    out = np.zeros((13, 768, n), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            batch = words[s:s + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True)
            hs = model(**enc, output_hidden_states=True).hidden_states
            # máscara dos subtokens reais: attention_mask sem [CLS]/[SEP]
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0                                   # [CLS]
            last = enc["attention_mask"].sum(dim=1) - 1      # posição do [SEP]
            mask[torch.arange(len(batch)), last] = 0
            m = mask.unsqueeze(-1).float()
            denom = m.sum(dim=1).clamp(min=1.0)              # (b, 1)
            for L in range(13):
                mean = (hs[L] * m).sum(dim=1) / denom        # (b, 768)
                out[L, :, s:s + len(batch)] = mean.T.numpy()
    return out


if CACHE_NPZ.exists():
    d = np.load(CACHE_NPZ, allow_pickle=False)
    pt_H, en_H = d["pt"], d["en"]
    assert list(d["pt_words"]) == pt_words and list(d["en_words"]) == en_words
    print("cache carregado:", CACHE_NPZ)
else:
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    print("extraindo PT...")
    pt_H = extract_mbert(pt_words, tokenizer, model)
    print("extraindo EN...")
    en_H = extract_mbert(en_words, tokenizer, model)
    np.savez_compressed(CACHE_NPZ, pt=pt_H, en=en_H,
                        pt_words=np.array(pt_words), en_words=np.array(en_words))
    print("cache salvo:", CACHE_NPZ)

# %% [markdown]
# ## Métricas por camada
#
# Para cada camada L: frame GSVD com 900/800 amostras da base (mesma
# semente do resto do paper), CKA, espectro de θ e retrieval P@1 nos 400
# pares de teste (centrados com as médias da base, cosseno entre os 400
# candidatos EN).

# %%
def p_at_1(PT_base, EN_base, PT_test, EN_test):
    """Retrieval de tradução por cosseno no espaço completo."""
    Pt = PT_test - PT_base.mean(axis=1, keepdims=True)
    Et = EN_test - EN_base.mean(axis=1, keepdims=True)
    Pt = Pt / np.maximum(np.linalg.norm(Pt, axis=0), 1e-12)
    Et = Et / np.maximum(np.linalg.norm(Et, axis=0), 1e-12)
    pred = np.argmax(Pt.T @ Et, axis=1)
    return float(np.mean(pred == np.arange(Pt.shape[1])))


def layer_metrics(PT_L, EN_L):
    PT_base, PT_test = PT_L[:, :N_BASE], PT_L[:, N_BASE:]
    EN_base, EN_test = EN_L[:, :N_BASE], EN_L[:, N_BASE:]
    g, A, B = build_frame(PT_base, EN_base, n_A=900, n_B=800, seed=SEED)
    th = get_angles_C(g.C)
    return {
        "cka": linear_cka(A, B),
        "std_theta": float(np.std(th)),
        "frac_shared": float(np.mean(np.abs(th - 45.0) <= 5.0)),
        "p_at_1": p_at_1(PT_base, EN_base, PT_test, EN_test),
    }


rows = []
for L in range(13):
    m = layer_metrics(pt_H[L].astype(float), en_H[L].astype(float))
    m["layer"] = L
    rows.append(m)
    print(f"camada {L:2d}: cka={m['cka']:.3f}  std={m['std_theta']:.2f}  "
          f"frac45={m['frac_shared']:.3f}  P@1={m['p_at_1']:.3f}")

# %% [markdown]
# ## Baseline MUSE
#
# Mesmas métricas no frame estático de `paper_common`: θ do frame pronto,
# CKA em `f.A`/`f.B`, e P@1 nos mesmos 400 pares de teste sobre os vetores
# MUSE (base = as mesmas 1200 palavras).

# %%
PT_muse_base = f.pt_X[:, [f.pt_idx[p] for e, p in base_pairs]]
EN_muse_base = f.en_X[:, [f.en_idx[e] for e, p in base_pairs]]
PT_muse_test = np.stack([f.v_pt(p) for e, p in test_pairs], axis=1)
EN_muse_test = np.stack([f.v_en(e) for e, p in test_pairs], axis=1)

muse = {
    "cka": linear_cka(f.A, f.B),
    "std_theta": float(np.std(f.angles)),
    "frac_shared": float(np.mean(np.abs(f.angles - 45.0) <= 5.0)),
    "p_at_1": p_at_1(PT_muse_base, EN_muse_base, PT_muse_test, EN_muse_test),
}
print("MUSE:", muse)

# %% [markdown]
# ## Figura — frac_shared e P@1 por camada, com o baseline MUSE

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

layers = [r["layer"] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, key, label in [(axes[0], "frac_shared", "fração |θ−45°| ≤ 5°"),
                       (axes[1], "p_at_1", "P@1 retrieval de tradução")]:
    ax.plot(layers, [r[key] for r in rows], "o-", label="mBERT (por camada)")
    ax.axhline(muse[key], color="gray", ls="--", label="MUSE (estático)")
    ax.set_xlabel("camada")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("Exp. 5 — quão interlingual é cada camada do mBERT?")
fig.tight_layout()
fig.savefig(RESULTS / "exp5_layers.png", dpi=150)
print("figura salva em", RESULTS / "exp5_layers.png")

# %% [markdown]
# ## Salvar resultados

# %%
best_frac = max(rows, key=lambda r: r["frac_shared"])["layer"]
best_p1 = max(rows, key=lambda r: r["p_at_1"])["layer"]
out = {"layers": rows, "muse_baseline": muse,
       "best_layer_frac_shared": best_frac, "best_layer_p_at_1": best_p1}
with open(RESULTS / "exp5.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)
print("exp5.json salvo. melhor camada (frac_shared):", best_frac,
      "| melhor camada (P@1):", best_p1)
