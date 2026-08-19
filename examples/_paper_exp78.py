# -*- coding: utf-8 -*-
# %% [markdown]
# # Experimentos 7 e 8 — traduzibilidade do espectro e entidades nos extremos
#
# Este script usa o frame GSVD conjunto PT/EN (ver `paper_common.py`). Cada
# direção $i$ de $H$ tem um ângulo $\theta_i \in [15°, 70°]$: $\theta$ baixo =
# específico do português, $\theta \approx 45°$ = compartilhado, $\theta$ alto =
# específico do inglês.
#
# - **Experimento 7:** mede a *coerência de tradução* de cada direção — quão
#   bem as top palavras PT e EN alinhadas à direção se traduzem entre si. A
#   hipótese é que a coerência tem pico na faixa compartilhada ($\theta$ perto
#   de 45°) e cai nos extremos, i.e., o espectro ordena as direções por
#   "traduzibilidade".
# - **Experimento 8:** testa se os extremos do espectro codificam *entidades*
#   (nomes próprios, cultura/corpus) em vez de vocabulário comum, usando como
#   proxy de entidade as palavras que não aparecem no dicionário de tradução.

# %%
import json

import matplotlib

matplotlib.use("Agg")  # no notebook: usar inline
import matplotlib.pyplot as plt
import numpy as np

from paper_common import Frame, RESULTS

SEED = 20260819
f = Frame()
angles = f.angles
print(f"frame pronto: {len(angles)} direções, θ em "
      f"[{angles.min():.1f}°, {angles.max():.1f}°]")

# %% [markdown]
# ## Experimento 7 — coerência de tradução por direção
#
# Para cada direção $i$: pegamos as top-10 palavras PT e top-10 EN mais
# alinhadas a $H_i$. A coerência é a média simétrica de duas frações:
# (i) fração das palavras PT cujo conjunto de traduções (`pt2en`) intersecta o
# top-10 EN; (ii) fração das EN cujo `en2pt` intersecta o top-10 PT.
# Palavras fora do dicionário contam como não-coerentes.

# %%
K = 10
n_dir = f.g.H.shape[0]
coherence = np.zeros(n_dir)
tops = []
for i in range(n_dir):
    tp, te = f.top_words(f.g.H[i, :], K)
    te_set, tp_set = set(te), set(tp)
    frac_pt = np.mean([bool(f.pt2en.get(w, set()) & te_set) for w in tp])
    frac_en = np.mean([bool(f.en2pt.get(w, set()) & tp_set) for w in te])
    coherence[i] = 0.5 * (frac_pt + frac_en)
    tops.append((tp, te))
print(f"coerência média global: {coherence.mean():.3f}")

# %% [markdown]
# ### (a) Scatter coerência × θ com curva binned (bins de 5°)

# %%
bin_edges = np.arange(15.0, 75.0, 5.0)
bin_centers, bin_means, bin_counts = [], [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    m = (angles >= lo) & (angles < hi)
    if m.sum() > 0:
        bin_centers.append(0.5 * (lo + hi))
        bin_means.append(float(coherence[m].mean()))
        bin_counts.append(int(m.sum()))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(angles, coherence, s=12, alpha=0.45, label="direções")
ax.plot(bin_centers, bin_means, "o-", color="crimson", lw=2,
        label="média por bin de 5°")
ax.axvline(45, color="gray", ls="--", lw=1)
ax.set_xlabel("θ da direção (graus)")
ax.set_ylabel("coerência de tradução")
ax.set_title("Exp. 7 — coerência de tradução × ângulo do espectro")
ax.legend()
fig.tight_layout()
fig.savefig("_paper_exp78_fig7.png", dpi=150)

# %% [markdown]
# ### (b) Pico e médias por faixa; (c) correlação com −|θ−45°|

# %%
peak_j = int(np.argmax(bin_means))
low_m = angles < 30
mid_m = (angles >= 40) & (angles <= 50)
high_m = angles > 60
mean_low = float(coherence[low_m].mean()) if low_m.any() else None
mean_mid = float(coherence[mid_m].mean()) if mid_m.any() else None
mean_high = float(coherence[high_m].mean()) if high_m.any() else None
corr = float(np.corrcoef(coherence, -np.abs(angles - 45.0))[0, 1])

print(f"pico: bin centrado em {bin_centers[peak_j]:.1f}° "
      f"(coerência média {bin_means[peak_j]:.3f}, n={bin_counts[peak_j]})")
print(f"média θ<30°:   {mean_low:.3f}  (n={int(low_m.sum())})")
print(f"média 40–50°:  {mean_mid:.3f}  (n={int(mid_m.sum())})")
print(f"média θ>60°:   {mean_high:.3f}  (n={int(high_m.sum())})")
print(f"corr(coerência, −|θ−45°|) = {corr:.3f}")

# %% [markdown]
# ### (d) Exemplos qualitativos: 3 direções de alta e 3 de baixa coerência

# %%
order = np.argsort(-coherence)
examples = {"alta": [], "baixa": []}
for label, idxs in [("alta", order[:3]), ("baixa", order[-3:])]:
    print(f"\n=== coerência {label} ===")
    for i in idxs:
        tp, te = tops[i]
        print(f"dir {i:3d}  θ={angles[i]:5.1f}°  coerência={coherence[i]:.2f}")
        print(f"   PT: {', '.join(tp)}")
        print(f"   EN: {', '.join(te)}")
        examples[label].append({
            "direction": int(i), "theta": float(angles[i]),
            "coherence": float(coherence[i]), "top_pt": tp, "top_en": te})

# %%
exp7 = {
    "seed": SEED,
    "k_top_words": K,
    "n_directions": int(n_dir),
    "binned_curve": {
        "bin_width_deg": 5.0,
        "centers": [float(c) for c in bin_centers],
        "mean_coherence": bin_means,
        "counts": bin_counts,
    },
    "peak": {"center_deg": float(bin_centers[peak_j]),
             "mean_coherence": float(bin_means[peak_j]),
             "count": bin_counts[peak_j]},
    "band_means": {"theta_lt_30": mean_low, "theta_40_50": mean_mid,
                   "theta_gt_60": mean_high},
    "corr_coherence_neg_abs_theta_minus_45": corr,
    "examples": examples,
}
with open(RESULTS / "exp7.json", "w", encoding="utf-8") as fh:
    json.dump(exp7, fh, ensure_ascii=False, indent=2)
print("salvo:", RESULTS / "exp7.json")

# %% [markdown]
# ## Experimento 8 — os extremos codificam entidades, não gramática
#
# Proxy de entidade: palavra do vocabulário (`isalpha`, `len>2`) que **não**
# aparece no dicionário de tradução (`pt2en` para PT, `en2pt` para EN) —
# tipicamente nomes próprios/entidades (o vocab MUSE é todo minúsculo).
# Palavras "comuns" são as que aparecem no dicionário. Para cada palavra
# medimos a fração de energia nas direções extremas do espectro
# (θ<30° ou θ>60°) e o ângulo θ da própria palavra.

# %%
rng = np.random.default_rng(SEED)
EXT_LO, EXT_HI = 30.0, 60.0
ext_mask = (angles < EXT_LO) | (angles > EXT_HI)
pt_ext_mask = angles < EXT_LO   # extremo "próprio" do PT
en_ext_mask = angles > EXT_HI   # extremo "próprio" do EN
print(f"direções extremas: {int(ext_mask.sum())} de {n_dir} "
      f"(PT<30°: {int(pt_ext_mask.sum())}, EN>60°: {int(en_ext_mask.sum())})")


def candidates(words, trans_dict):
    ent = [(j, w) for j, w in enumerate(words)
           if w.isalpha() and len(w) > 2 and w not in trans_dict]
    com = [(j, w) for j, w in enumerate(words)
           if w.isalpha() and len(w) > 2 and w in trans_dict]
    return ent, com


def sample_top(cands, n, rng, pool=5000):
    """Amostra n itens entre os `pool` mais frequentes (menor índice)."""
    cands = sorted(cands)[:pool]
    sel = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [cands[s] for s in sel]


def sample_matched(ent, com, n, rng, n_bins=10):
    """Amostra casando a distribuição de frequência: mesmos decis de índice
    no vocabulário para entidades e comuns."""
    all_idx = np.array([j for j, _ in ent] + [j for j, _ in com])
    edges = np.quantile(all_idx, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1
    per_bin = n // n_bins
    out_e, out_c = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        eb = [x for x in ent if lo <= x[0] < hi]
        cb = [x for x in com if lo <= x[0] < hi]
        m = min(per_bin, len(eb), len(cb))
        if m == 0:
            continue
        out_e += [eb[s] for s in rng.choice(len(eb), size=m, replace=False)]
        out_c += [cb[s] for s in rng.choice(len(cb), size=m, replace=False)]
    return out_e, out_c


def measure(sample, X, own_mask):
    """Para cada (idx, palavra): energia nos extremos, energia no extremo da
    própria língua e θ da palavra."""
    ext_e, own_e, thetas, words = [], [], [], []
    for j, w in sample:
        v = X[:, j]
        prof = f.energy_profile(v)
        ext_e.append(float(prof[ext_mask].sum()))
        own_e.append(float(prof[own_mask].sum()))
        thetas.append(float(np.ravel(f.theta(v))[0]))
        words.append(w)
    return (np.array(ext_e), np.array(own_e), np.array(thetas), words)


def welch(a, b):
    """t de Welch e p bicaudal (aprox. normal, n grande)."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p)


N = 500
pt_ent_c, pt_com_c = candidates(f.pt_words, f.pt2en)
en_ent_c, en_com_c = candidates(f.en_words, f.en2pt)
print(f"candidatos PT: {len(pt_ent_c)} entidades, {len(pt_com_c)} comuns")
print(f"candidatos EN: {len(en_ent_c)} entidades, {len(en_com_c)} comuns")

samples = {
    "pt": {"ent": sample_top(pt_ent_c, N, rng), "com": sample_top(pt_com_c, N, rng)},
    "en": {"ent": sample_top(en_ent_c, N, rng), "com": sample_top(en_com_c, N, rng)},
}
m_pt_e, m_pt_c = sample_matched(pt_ent_c, pt_com_c, N, rng)
m_en_e, m_en_c = sample_matched(en_ent_c, en_com_c, N, rng)
matched = {"pt": {"ent": m_pt_e, "com": m_pt_c},
           "en": {"ent": m_en_e, "com": m_en_c}}

# %% [markdown]
# ### (a) Energia nos extremos: entidades vs comuns; (b) θ médio por grupo

# %%
def run_block(samp):
    res = {}
    for lang, X, own in [("pt", f.pt_X, pt_ext_mask), ("en", f.en_X, en_ext_mask)]:
        out = {}
        for grp in ["ent", "com"]:
            ext_e, own_e, th, words = measure(samp[lang][grp], X, own)
            out[grp] = {"n": len(words),
                        "extreme_energy_mean": float(ext_e.mean()),
                        "extreme_energy_std": float(ext_e.std(ddof=1)),
                        "own_extreme_energy_mean": float(own_e.mean()),
                        "theta_mean": float(th.mean()),
                        "theta_std": float(th.std(ddof=1)),
                        "_raw": (ext_e, own_e, th, words)}
        t, p = welch(out["ent"]["_raw"][0], out["com"]["_raw"][0])
        out["welch_t_extreme_energy_ent_vs_com"] = t
        out["welch_p"] = p
        res[lang] = out
    return res


res_main = run_block(samples)
res_matched = run_block(matched)

for name, res in [("amostra principal (top-freq)", res_main),
                  ("amostra casada por decil de frequência", res_matched)]:
    print(f"\n===== {name} =====")
    for lang in ["pt", "en"]:
        r = res[lang]
        print(f"[{lang.upper()}] energia nos extremos — "
              f"entidades {r['ent']['extreme_energy_mean']:.3f}"
              f"±{r['ent']['extreme_energy_std']:.3f}  vs  "
              f"comuns {r['com']['extreme_energy_mean']:.3f}"
              f"±{r['com']['extreme_energy_std']:.3f}   "
              f"(Welch t={r['welch_t_extreme_energy_ent_vs_com']:.2f}, "
              f"p={r['welch_p']:.2e})")
        print(f"      θ médio — entidades {r['ent']['theta_mean']:.2f}° "
              f"vs comuns {r['com']['theta_mean']:.2f}°")

# %% [markdown]
# ### (c) Verificação qualitativa: top-15 palavras com mais energia no
# extremo da própria língua — esperamos majoritariamente nomes/entidades.

# %%
qual = {}
for lang in ["pt", "en"]:
    ext_e_ent, own_ent, th_ent, w_ent = res_main[lang]["ent"]["_raw"]
    ext_e_com, own_com, th_com, w_com = res_main[lang]["com"]["_raw"]
    own_all = np.concatenate([own_ent, own_com])
    w_all = w_ent + w_com
    grp_all = ["entidade"] * len(w_ent) + ["comum"] * len(w_com)
    top = np.argsort(-own_all)[:15]
    qual[lang] = [{"word": w_all[j], "group": grp_all[j],
                   "own_extreme_energy": float(own_all[j])} for j in top]
    print(f"\n[{lang.upper()}] top-15 energia no extremo da própria língua:")
    for d in qual[lang]:
        print(f"   {d['word']:<20s} {d['group']:<9s} {d['own_extreme_energy']:.3f}")

# %% [markdown]
# ### (d) Figura: distribuições de energia nos extremos, entidades vs comuns

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, lang in zip(axes, ["pt", "en"]):
    e = res_main[lang]["ent"]["_raw"][0]
    c = res_main[lang]["com"]["_raw"][0]
    bins = np.linspace(0, max(e.max(), c.max()), 30)
    ax.hist(c, bins=bins, alpha=0.6, label="comuns", density=True)
    ax.hist(e, bins=bins, alpha=0.6, label="entidades", density=True)
    ax.set_title(f"{lang.upper()} — energia em θ<30° ou θ>60°")
    ax.set_xlabel("fração de energia nos extremos")
    ax.legend()
axes[0].set_ylabel("densidade")
fig.suptitle("Exp. 8 — entidades concentram mais energia nos extremos")
fig.tight_layout()
fig.savefig("_paper_exp78_fig8.png", dpi=150)

# %%
def clean(res):
    out = {}
    for lang, r in res.items():
        out[lang] = {g: {k: v for k, v in r[g].items() if k != "_raw"}
                     for g in ["ent", "com"]}
        out[lang]["welch_t_extreme_energy_ent_vs_com"] = \
            r["welch_t_extreme_energy_ent_vs_com"]
        out[lang]["welch_p"] = r["welch_p"]
    return out


exp8 = {
    "seed": SEED,
    "n_per_group": N,
    "extreme_definition_deg": {"low": EXT_LO, "high": EXT_HI},
    "n_extreme_directions": int(ext_mask.sum()),
    "main_sample": clean(res_main),
    "frequency_matched_sample": clean(res_matched),
    "top15_own_extreme": qual,
}
with open(RESULTS / "exp8.json", "w", encoding="utf-8") as fh:
    json.dump(exp8, fh, ensure_ascii=False, indent=2)
print("salvo:", RESULTS / "exp8.json")
