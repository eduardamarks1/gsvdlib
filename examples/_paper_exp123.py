# -*- coding: utf-8 -*-
# %% [markdown]
# # Experimentos 1–3: analogias cross-língue, band-pass funcional e direções centrais
#
# Este script usa o frame GSVD conjunto PT/EN (ver `paper_common.py`). Cada
# direção `i` de `H` tem um ângulo de alinhamento `angles[i]` em [15°, 70°]:
# ângulos baixos indicam direções específicas do português, θ≈45° direções
# compartilhadas pelas duas línguas, e ângulos altos direções específicas do
# inglês. Testamos três hipóteses funcionais sobre esse espectro:
#
# 1. **Exp. 1** — offsets de analogia extraídos em PT transferem para EN, e
#    transferem melhor quando concentram energia na banda compartilhada.
# 2. **Exp. 2** — identidade de língua vive nos extremos do espectro;
#    semântica/tradução vive na banda compartilhada.
# 3. **Exp. 3** — o retrieval de tradução satura com poucas direções centrais,
#    superando o mesmo número de direções aleatórias.

# %%
import json

import matplotlib

matplotlib.use("Agg")  # no notebook, o backend inline substitui isto
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from paper_common import Frame, RESULTS, nn_search

SEED = 1234
rng = np.random.default_rng(SEED)

f = Frame(seed=SEED)
print(f"Frame pronto: {len(f.pairs)} pares, {len(f.test_pairs)} test_pairs")
print(f"Ângulos: min={f.angles.min():.1f}°, max={f.angles.max():.1f}°")

# %% [markdown]
# ## Experimento 1 — Transferência de analogias entre línguas
#
# Datasets de relações curados manualmente (capital–país e
# masculino–feminino) e gerados automaticamente (singular–plural, palavras
# `w` frequentes tais que `w` e `w+"s"` estão no vocabulário). Cada par é
# verificado contra o vocabulário; reportamos quantos sobram.
#
# **Protocolo cross-língue**: para o par PT `(p_a, p_b)` formamos o offset
# `r_pt = v_pt(p_a) − v_pt(p_b)`; para um par EN `(e_a, e_b)` da mesma
# categoria, a consulta é `v_en(e_b) + r_pt` e a resposta correta é `e_a`
# (ex.: `v_en(france) + [v_pt(paris) − v_pt(frança)] ≈ v_en(paris)`).
# Busca por cosseno nos 20k vetores EN, excluindo `e_b`.

# %%
CAP_PT = [("paris", "frança"), ("londres", "inglaterra"), ("roma", "itália"),
          ("berlim", "alemanha"), ("lisboa", "portugal"), ("moscou", "rússia"),
          ("atenas", "grécia"), ("madrid", "espanha"), ("viena", "áustria"),
          ("dublin", "irlanda"), ("oslo", "noruega"), ("estocolmo", "suécia"),
          ("varsóvia", "polônia"), ("budapeste", "hungria"),
          ("praga", "tchecoslováquia"), ("cairo", "egito"),
          ("tóquio", "japão"), ("pequim", "china"), ("brasília", "brasil"),
          ("ottawa", "canadá")]
CAP_EN = [("paris", "france"), ("london", "england"), ("rome", "italy"),
          ("berlin", "germany"), ("lisbon", "portugal"), ("moscow", "russia"),
          ("athens", "greece"), ("madrid", "spain"), ("vienna", "austria"),
          ("dublin", "ireland"), ("oslo", "norway"), ("stockholm", "sweden"),
          ("warsaw", "poland"), ("budapest", "hungary"),
          ("prague", "czechoslovakia"), ("cairo", "egypt"),
          ("tokyo", "japan"), ("beijing", "china"), ("ottawa", "canada")]
GEN_PT = [("rei", "rainha"), ("homem", "mulher"), ("pai", "mãe"),
          ("ator", "atriz"), ("irmão", "irmã"), ("filho", "filha"),
          ("príncipe", "princesa"), ("tio", "tia"), ("avô", "avó"),
          ("menino", "menina")]
GEN_EN = [("king", "queen"), ("man", "woman"), ("father", "mother"),
          ("actor", "actress"), ("brother", "sister"), ("son", "daughter"),
          ("prince", "princess"), ("uncle", "aunt"),
          ("grandfather", "grandmother"), ("boy", "girl")]


def filter_pairs(pairs, idx):
    return [(a, b) for a, b in pairs if a in idx and b in idx]


def plural_pairs(idx, n=25):
    """Pares (plural, singular) — offset 'adicionar s' — mais frequentes."""
    cands = []
    for w, i in idx.items():
        if w.isalpha() and len(w) > 2 and w + "s" in idx:
            cands.append((i, (w + "s", w)))
    cands.sort()
    return [p for _, p in cands[:n]]


datasets = {
    "capital-pais": (filter_pairs(CAP_PT, f.pt_idx), filter_pairs(CAP_EN, f.en_idx)),
    "masc-fem": (filter_pairs(GEN_PT, f.pt_idx), filter_pairs(GEN_EN, f.en_idx)),
    "sing-plural": (plural_pairs(f.pt_idx), plural_pairs(f.en_idx)),
}
for cat, (ppt, pen) in datasets.items():
    print(f"{cat}: {len(ppt)} pares PT, {len(pen)} pares EN no vocab")

# %%
shared_mask = f.band_mask(40, 50)


def run_analogy(pt_pairs, en_pairs, max_combos=400, project_mask=None,
                seed=SEED):
    """Analogia cross-língue. Retorna lista de resultados por combinação.

    Cada item: (i_pt, hit1, hit5) onde i_pt indexa o offset PT usado.
    """
    r = np.random.default_rng(seed)
    combos = [(i, j) for i in range(len(pt_pairs))
              for j in range(len(en_pairs))
              if pt_pairs[i] != en_pairs[j]]
    if len(combos) > max_combos:
        combos = [combos[t] for t in r.choice(len(combos), max_combos,
                                              replace=False)]
    X_en = f.en_X
    offsets = {i: f.v_pt(a) - f.v_pt(b) for i, (a, b) in enumerate(pt_pairs)}
    if project_mask is not None:
        X_en = f.band_project(f.en_X, project_mask)
        offsets = {i: f.band_project(v.reshape(-1, 1), project_mask).ravel()
                   for i, v in offsets.items()}
    out = []
    for i, j in combos:
        e_a, e_b = en_pairs[j]
        q = (X_en[:, f.en_idx[e_b]] + offsets[i])
        top = nn_search(q, X_en, exclude={f.en_idx[e_b]}, k=5)
        gold = f.en_idx[e_a]
        out.append((i, int(top[0] == gold), int(gold in top)))
    return out


exp1 = {}
for cat, (ppt, pen) in datasets.items():
    res = run_analogy(ppt, pen)
    top1 = float(np.mean([h1 for _, h1, _ in res]))
    top5 = float(np.mean([h5 for _, _, h5 in res]))
    exp1[cat] = {"n_pt": len(ppt), "n_en": len(pen), "n_combos": len(res),
                 "top1": top1, "top5": top5, "_res": res, "_ppt": ppt}
    print(f"{cat}: top1={top1:.3f}, top5={top5:.3f} ({len(res)} combos)")

# %% [markdown]
# ### Energia na banda compartilhada × taxa de acerto do offset
#
# Para cada offset PT `r_pt`, medimos a fração da sua energia (no frame H)
# que cai na banda compartilhada 40°–50°, e correlacionamos (Pearson) com a
# acurácia média top-1 das combinações que usam aquele offset. Também
# reportamos as médias por quartil de energia.

# %%
band_energy_corr = {}
all_e, all_a = [], []
for cat, d in exp1.items():
    per_offset = {}
    for i, h1, _ in d["_res"]:
        per_offset.setdefault(i, []).append(h1)
    energies, accs = [], []
    for i, hits in sorted(per_offset.items()):
        p_a, p_b = d["_ppt"][i]
        r_pt = f.v_pt(p_a) - f.v_pt(p_b)
        e = float(f.energy_profile(r_pt)[shared_mask].sum())
        energies.append(e)
        accs.append(float(np.mean(hits)))
    energies, accs = np.array(energies), np.array(accs)
    all_e.append(energies)
    all_a.append(accs)
    if len(energies) > 2 and energies.std() > 0 and accs.std() > 0:
        pear = float(np.corrcoef(energies, accs)[0, 1])
    else:
        pear = None
    # quartis por energia
    order = np.argsort(energies)
    qs = np.array_split(order, 4)
    quart = [{"energia_media": float(energies[q].mean()),
              "acc_media": float(accs[q].mean())} for q in qs]
    band_energy_corr[cat] = {
        "pearson": pear, "quartis": quart,
        "energia_por_offset": energies.round(4).tolist(),
        "acc_por_offset": accs.round(4).tolist()}
    print(f"{cat}: pearson(energia_banda, acc)={pear}")
    for k, q in enumerate(quart):
        print(f"  Q{k+1}: energia={q['energia_media']:.3f} "
              f"acc={q['acc_media']:.3f}")

# correlação global (todas as categorias juntas)
Eg, Ag = np.concatenate(all_e), np.concatenate(all_a)
pear_global = float(np.corrcoef(Eg, Ag)[0, 1])
print(f"pearson global = {pear_global:.3f}")

exp1_out = {cat: {k: v for k, v in d.items() if not k.startswith("_")}
            for cat, d in exp1.items()}
exp1_out["banda_compartilhada_40_50"] = band_energy_corr
exp1_out["pearson_global"] = pear_global
with open(RESULTS / "exp1.json", "w", encoding="utf-8") as fh:
    json.dump(exp1_out, fh, ensure_ascii=False, indent=2)
print("salvo:", RESULTS / "exp1.json")

# %% [markdown]
# ## Experimento 2 — Band-pass funcional
#
# Quatro bandas: **PT-side** (θ<40°), **shared** (40°≤θ≤50°), **EN-side**
# (θ>50°) e um **controle** com o mesmo número de direções da shared, mas
# sorteadas ao acaso (semente fixa). Para cada banda avaliamos três tarefas
# sobre vetores band-projetados:
#
# 1. **Identificação de língua** — regressão logística PT vs EN nas 2000
#    palavras dos `test_pairs` (70/30);
# 2. **Retrieval de tradução** — P@1/P@5 do vetor PT projetado contra os
#    1000 candidatos EN projetados (500 consultas);
# 3. **Analogia capital–país** — protocolo do Exp. 1 (200 combinações).

# %%
masks = {
    "pt_side": f.angles < 40,
    "shared": shared_mask,
    "en_side": f.angles > 50,
}
rng2 = np.random.default_rng(SEED)
rand_dirs = rng2.choice(300, int(shared_mask.sum()), replace=False)
ctrl = np.zeros(300, dtype=bool)
ctrl[rand_dirs] = True
masks["controle_aleatorio"] = ctrl
for name, m in masks.items():
    print(f"{name}: {int(m.sum())} direções")

# matrizes dos test_pairs (colunas alinhadas: i-ésimo PT traduz i-ésimo EN)
tp_en = [e for e, p in f.test_pairs]
tp_pt = [p for e, p in f.test_pairs]
PT_tp = f.pt_X[:, [f.pt_idx[w] for w in tp_pt]]
EN_tp = f.en_X[:, [f.en_idx[w] for w in tp_en]]

ret_idx = np.random.default_rng(SEED).choice(len(f.test_pairs), 500,
                                             replace=False)


def retrieval_p1p5(PTb, ENb, queries=ret_idx):
    """P@1/P@5: consulta PT projetada, candidatos = 1000 EN projetados."""
    ENn = ENb / np.maximum(np.linalg.norm(ENb, axis=0), 1e-12)
    hits1 = hits5 = 0
    for i in queries:
        q = PTb[:, i]
        q = q / max(np.linalg.norm(q), 1e-12)
        top = np.argsort(-(q @ ENn))[:5]
        hits1 += int(top[0] == i)
        hits5 += int(i in top)
    n = len(queries)
    return hits1 / n, hits5 / n


def language_id_acc(PTb, ENb):
    X = np.hstack([PTb, ENb]).T
    y = np.r_[np.zeros(PTb.shape[1]), np.ones(ENb.shape[1])]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


exp2 = {}
cap_pt, cap_en = datasets["capital-pais"]
for name, m in masks.items():
    PTb = f.band_project(PT_tp, m)
    ENb = f.band_project(EN_tp, m)
    lang = language_id_acc(PTb, ENb)
    p1, p5 = retrieval_p1p5(PTb, ENb)
    ana = run_analogy(cap_pt, cap_en, max_combos=200, project_mask=m)
    a1 = float(np.mean([h for _, h, _ in ana]))
    a5 = float(np.mean([h for _, _, h in ana]))
    exp2[name] = {"n_direcoes": int(m.sum()), "lang_id_acc": lang,
                  "trad_P1": p1, "trad_P5": p5,
                  "analogia_top1": a1, "analogia_top5": a5}
    print(f"{name:20s} lang={lang:.3f} P@1={p1:.3f} P@5={p5:.3f} "
          f"ana1={a1:.3f} ana5={a5:.3f}")

with open(RESULTS / "exp2.json", "w", encoding="utf-8") as fh:
    json.dump(exp2, fh, ensure_ascii=False, indent=2)
print("salvo:", RESULTS / "exp2.json")

# %% [markdown]
# ## Experimento 3 — Retrieval de tradução × número de direções centrais
#
# Ordenamos as 300 direções por |θ−45°| crescente (mais "compartilhadas"
# primeiro). Para cada `k`, mantemos as `k` mais centrais e medimos P@1/P@5
# de tradução (protocolo do Exp. 2-ii). Baseline: `k` direções aleatórias,
# média de 3 sementes.

# %%
central_order = np.argsort(np.abs(f.angles - 45.0))
ks = [10, 25, 50, 75, 100, 150, 200, 300]
exp3 = {"ks": ks, "central": [], "aleatorio": []}
for k in ks:
    m = np.zeros(300, dtype=bool)
    m[central_order[:k]] = True
    p1, p5 = retrieval_p1p5(f.band_project(PT_tp, m), f.band_project(EN_tp, m))
    exp3["central"].append({"k": k, "P1": p1, "P5": p5})
    r1s, r5s = [], []
    for s in range(3):
        rm = np.zeros(300, dtype=bool)
        rm[np.random.default_rng(SEED + s).choice(300, k, replace=False)] = True
        rp1, rp5 = retrieval_p1p5(f.band_project(PT_tp, rm),
                                  f.band_project(EN_tp, rm))
        r1s.append(rp1)
        r5s.append(rp5)
    exp3["aleatorio"].append({"k": k, "P1": float(np.mean(r1s)),
                              "P5": float(np.mean(r5s))})
    print(f"k={k:3d} central P@1={p1:.3f} P@5={p5:.3f} | "
          f"aleatório P@1={np.mean(r1s):.3f} P@5={np.mean(r5s):.3f}")

with open(RESULTS / "exp3.json", "w", encoding="utf-8") as fh:
    json.dump(exp3, fh, ensure_ascii=False, indent=2)
print("salvo:", RESULTS / "exp3.json")

# %% [markdown]
# ### Curva P@1 × k — direções centrais vs aleatórias

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ks, [d["P1"] for d in exp3["central"]], "o-",
        label="k mais centrais (|θ−45°|)")
ax.plot(ks, [d["P1"] for d in exp3["aleatorio"]], "s--",
        label="k aleatórias (média de 3 sementes)")
ax.set_xlabel("k (número de direções mantidas)")
ax.set_ylabel("P@1 de tradução")
ax.set_title("Retrieval de tradução vs direções centrais do frame GSVD")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

# %% [markdown]
# **Leitura esperada**: poucas direções centrais já recuperam quase todo o
# desempenho de tradução, enquanto direções aleatórias precisam de muito
# mais — evidência de que a semântica compartilhada se concentra na banda
# central do espectro de alinhamento.
