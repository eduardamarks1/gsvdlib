# -*- coding: utf-8 -*-
# %% [markdown]
# # Experimentos 4, 6 e 9: proximidade de línguas, controle não-alinhado e diminutivos
#
# Três experimentos complementares sobre o espectro de alinhamento GSVD
# (ver `paper_common.py`):
#
# 4. **Largura da banda compartilhada × proximidade das línguas** — pares
#    PT×EN, PT×ES, EN×ES e EN×DE, mesmo protocolo de frame; hipótese:
#    línguas irmãs (PT×ES) têm banda compartilhada mais larga.
# 6. **Controle negativo** — embeddings CommonCrawl PT e EN treinados em
#    espaços independentes (não-alinhados): o método deve detectar a
#    ausência de estrutura compartilhada.
# 9. **Diminutivos em texto informal** — o offset -inho/-inha nos vetores
#    CommonCrawl (via Procrustes para o espaço MUSE) vs Wikipédia.

# %%
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no notebook, o backend inline substitui isto
import matplotlib.pyplot as plt
import numpy as np

from gsvdlib import get_angles_C, get_most_similar_row_H, linear_cka
from paper_common import CACHE, RESULTS, Frame, build_frame, load_muse_pair

CC = Path.home() / ".gsvdlib" / "cc"
SEED = 1234


def load_words_X(path):
    d = np.load(path)
    return [str(w) for w in d["words"]], d["X"].astype(float)


def load_dict_txt(path):
    """Dicionário MUSE em texto: duas colunas separadas por espaço."""
    src, tgt = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                src.append(parts[0])
                tgt.append(parts[1])
    return src, tgt


def build_pairs_generic(words_a, words_b, dict_a, dict_b):
    """Mesma filtragem de build_pairs, para qualquer par de vocabulários.

    dict_a[i] traduz dict_b[i]; devolve pares (w_a, w_b) usáveis.
    """
    idx_a = {w: i for i, w in enumerate(words_a)}
    idx_b = {w: i for i, w in enumerate(words_b)}
    pairs, seen_a, seen_b = [], set(), set()
    for wa, wb in zip(dict_a, dict_b):
        if (wa in idx_a and wb in idx_b
                and wa.isalpha() and wb.isalpha()
                and len(wa) > 2 and len(wb) > 2
                and wa != wb
                and wa not in seen_a and wb not in seen_b):
            pairs.append((wa, wb))
            seen_a.add(wa)
            seen_b.add(wb)
    return pairs, idx_a, idx_b


def frame_means(A_feat, B_feat, n_A=900, n_B=800, seed=SEED):
    """Reproduz a amostragem de build_frame e devolve as médias da base."""
    rng = np.random.default_rng(seed)
    mA = A_feat[:, rng.permutation(A_feat.shape[1])[:n_A]].mean(
        axis=1, keepdims=True)
    mB = B_feat[:, rng.permutation(B_feat.shape[1])[:n_B]].mean(
        axis=1, keepdims=True)
    return mA, mB


def pair_metrics(name, Xa, Xb, idx_a, idx_b, pairs, seed=SEED,
                 n_train=1200, n_test=400):
    """Protocolo padrão: frame 900/800 de 1200 pares + retrieval em 400.

    Retorna dict de métricas e também (g, angles) para uso posterior.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    train = [pairs[i] for i in order[:n_train]]
    test = [pairs[i] for i in order[n_train:n_train + n_test]]
    A_all = Xa[:, [idx_a[wa] for wa, _ in train]]
    B_all = Xb[:, [idx_b[wb] for _, wb in train]]
    g, A, B = build_frame(A_all, B_all, seed=seed)
    angles = get_angles_C(g.C)
    cka = linear_cka(A, B)
    band = float(np.mean(np.abs(angles - 45.0) <= 5.0))

    # retrieval de tradução: consulta lado A -> candidatos = os 400 do teste
    mA, mB = frame_means(A_all, B_all, seed=seed)
    Q = Xa[:, [idx_a[wa] for wa, _ in test]] - mA
    Cand = Xb[:, [idx_b[wb] for _, wb in test]] - mB
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=0), 1e-12)
    Cn = Cand / np.maximum(np.linalg.norm(Cand, axis=0), 1e-12)
    p1 = float(np.mean(np.argmax(Qn.T @ Cn, axis=1) == np.arange(len(test))))

    out = {"n_pares_vocab": len(pairs), "cka": round(cka, 4),
           "std_theta": round(float(angles.std()), 3),
           "frac_banda_45pm5": round(band, 4), "P1_traducao": round(p1, 4)}
    print(f"{name:8s} pares={len(pairs):5d} CKA={cka:.3f} "
          f"std(θ)={angles.std():.2f}° banda={band:.3f} P@1={p1:.3f}")
    return out, g, angles

# %% [markdown]
# ## Experimento 4 — Largura da banda compartilhada × proximidade das línguas
#
# Quatro pares de línguas, todos com vetores fastText alinhados no espaço
# MUSE multilíngue (wiki), mesmo protocolo: vocabulário pareado pelo
# dicionário MUSE (filtragem idêntica a `build_pairs`), 1200 pares de
# treino (semente 1234), frame 900/800 e métricas de largura de banda:
#
# - **CKA linear** entre as matrizes centradas do frame;
# - **desvio-padrão de θ** (espectro mais estreito ⇒ menos direções
#   específicas de língua);
# - **fração de direções com |θ−45°| ≤ 5°** (a banda compartilhada);
# - **P@1 de tradução** em 400 pares held-out (cosseno, vetores
#   centralizados com as médias da base do frame, candidatos = os 400).
#
# **Hipótese**: PT×ES (línguas irmãs) tem banda mais larga que EN×DE, com
# PT×EN e EN×ES intermediários. Os números são reportados como saírem.

# %%
pt_words, pt_X, en_words, en_X, dict_en, dict_pt = load_muse_pair()
es_words, es_X = load_words_X(CACHE / "wiki_multi_es_top20k.npz")
de_words, de_X = load_words_X(CACHE / "wiki_multi_de_top20k.npz")

d_pt_es = load_dict_txt(CACHE / "dict_pt-es.txt")   # pt -> es
d_en_es = load_dict_txt(CACHE / "dict_en-es.txt")   # en -> es
d_en_de = load_dict_txt(CACHE / "dict_en-de.txt")   # en -> de

lang_pairs = {
    "PTxEN": (pt_X, en_X, build_pairs_generic(pt_words, en_words,
                                              dict_pt, dict_en)),
    "PTxES": (pt_X, es_X, build_pairs_generic(pt_words, es_words,
                                              d_pt_es[0], d_pt_es[1])),
    "ENxES": (en_X, es_X, build_pairs_generic(en_words, es_words,
                                              d_en_es[0], d_en_es[1])),
    "ENxDE": (en_X, de_X, build_pairs_generic(en_words, de_words,
                                              d_en_de[0], d_en_de[1])),
}

exp4 = {}
angles_by_pair = {}
for name, (Xa, Xb, (pairs, idx_a, idx_b)) in lang_pairs.items():
    exp4[name], _, angles_by_pair[name] = pair_metrics(
        name, Xa, Xb, idx_a, idx_b, pairs)

# %% [markdown]
# ### Tabela final (ordenada por fração na banda compartilhada)

# %%
order4 = sorted(exp4, key=lambda k: -exp4[k]["frac_banda_45pm5"])
print(f"{'par':8s} {'CKA':>6s} {'std θ':>7s} {'banda':>7s} {'P@1':>6s}")
for k in order4:
    d = exp4[k]
    print(f"{k:8s} {d['cka']:6.3f} {d['std_theta']:7.2f} "
          f"{d['frac_banda_45pm5']:7.3f} {d['P1_traducao']:6.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
for k in order4:
    ax.hist(angles_by_pair[k], bins=40, histtype="step", label=k)
ax.axvline(45, color="gray", ls=":")
ax.set_xlabel("θ (graus)")
ax.set_ylabel("nº de direções")
ax.set_title("Espectro de alinhamento por par de línguas")
ax.legend()
fig.tight_layout()

exp4_out = {"ordenacao_por_banda": order4, "pares": exp4}
with open(RESULTS / "exp4.json", "w", encoding="utf-8") as fh:
    json.dump(exp4_out, fh, ensure_ascii=False, indent=2)
print("salvo:", RESULTS / "exp4.json")

# %% [markdown]
# ## Experimento 6 — Controle: embeddings não-alinhados
#
# Vetores fastText CommonCrawl PT e EN, treinados de forma independente
# (nenhum alinhamento entre os espaços). Pareamos o vocabulário com o
# mesmo dicionário MUSE do cache pt/en e aplicamos o protocolo do Exp. 4.
# Adicionalmente, medimos a **coerência de tradução das direções**: para
# 60 direções amostradas uniformemente ao longo do espectro de θ, tomamos
# as top-10 palavras de cada lado (cosseno contra o vocabulário) e a
# fração de palavras EN cuja tradução aparece no top-10 PT.
#
# **Predição**: sem alinhamento, o espectro perde sentido semântico — a
# coerência fica perto de zero em toda parte e o P@1 desaba para o nível
# do acaso (1/400 = 0.0025). Comparação: o par MUSE PT×EN alinhado.

# %%
cc_pt_words, cc_pt_X = load_words_X(CC / "cc_pt_top20k.npz")
cc_en_words, cc_en_X = load_words_X(CC / "cc_en_top20k.npz")

# dicionário como dict de sets (para a coerência)
en2pt = {}
for e, p in zip(dict_en, dict_pt):
    en2pt.setdefault(e, set()).add(p)

cc_pairs, cc_idx_pt, cc_idx_en = build_pairs_generic(
    cc_pt_words, cc_en_words, dict_pt, dict_en)
exp6_cc, g_cc, ang_cc = pair_metrics(
    "CC(des)", cc_pt_X, cc_en_X, cc_idx_pt, cc_idx_en, cc_pairs)

f = Frame(seed=SEED)
print(f"Frame MUSE alinhado: {len(f.pairs)} pares")


def direction_coherence(g, angles, X_pt, words_pt, X_en, words_en,
                        n_dirs=60, k=10):
    """Coerência de tradução por direção, amostrada uniformemente em θ."""
    order = np.argsort(angles)
    sel = order[np.linspace(0, len(order) - 1, n_dirs).astype(int)]
    out = []
    for i in sel:
        d = g.H[i]
        sp = get_most_similar_row_H(d, X_pt.T)
        se = get_most_similar_row_H(d, X_en.T)
        top_pt = {words_pt[j] for j in np.argsort(-np.abs(sp))[:k]}
        top_en = [words_en[j] for j in np.argsort(-np.abs(se))[:k]]
        frac = sum(1 for e in top_en if en2pt.get(e, set()) & top_pt) / k
        out.append((float(angles[i]), frac))
    return out


coh_cc = direction_coherence(g_cc, ang_cc, cc_pt_X, cc_pt_words,
                             cc_en_X, cc_en_words)
coh_muse = direction_coherence(f.g, f.angles, f.pt_X, f.pt_words,
                               f.en_X, f.en_words)

# métricas do par alinhado no mesmo formato (frame padrão + retrieval)
exp6_muse, _, _ = pair_metrics(
    "MUSE(al)", f.pt_X, f.en_X, f.pt_idx, f.en_idx,
    [(p, e) for e, p in f.pairs])

coh_cc_mean = float(np.mean([c for _, c in coh_cc]))
coh_muse_mean = float(np.mean([c for _, c in coh_muse]))
print(f"coerência média das direções: CC={coh_cc_mean:.3f} "
      f"MUSE={coh_muse_mean:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(*zip(*coh_muse), "o-", label="MUSE alinhado", alpha=0.7)
ax.plot(*zip(*coh_cc), "s--", label="CommonCrawl não-alinhado", alpha=0.7)
ax.set_xlabel("θ da direção (graus)")
ax.set_ylabel("coerência de tradução (top-10)")
ax.set_title("Coerência das direções: alinhado vs não-alinhado")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

exp6_out = {
    "nao_alinhado_cc": {**exp6_cc, "coerencia_media": round(coh_cc_mean, 4)},
    "alinhado_muse": {**exp6_muse,
                      "coerencia_media": round(coh_muse_mean, 4)},
    "coerencia_por_direcao_cc": [[round(a, 2), c] for a, c in coh_cc],
    "coerencia_por_direcao_muse": [[round(a, 2), c] for a, c in coh_muse],
    "acaso_P1": 1.0 / 400,
}
with open(RESULTS / "exp6.json", "w", encoding="utf-8") as fh:
    json.dump(exp6_out, fh, ensure_ascii=False, indent=2)
print("salvo:", RESULTS / "exp6.json")

# %% [markdown]
# ## Experimento 9 — Diminutivos em texto informal via Procrustes
#
# Nos vetores Wikipédia, o offset de diminutivo (-inho/-inha) tem coerência
# interna fraca (~0.14). Hipótese: no CommonCrawl (texto informal) a
# morfologia é produtiva e o diminutivo vira direção linear.
#
# **(a)** Alinhamos cc_pt ao espaço MUSE wiki.pt por Procrustes ortogonal:
# vocabulário comum entre cc_pt top-20k e o lado PT do cache MUSE,
# `W = U Vᵀ` da SVD de `B Aᵀ` (A=cc, B=muse, vetores como estão), que
# minimiza `‖W A − B‖_F` com W ortogonal. Validação em palavras held-out.

# %%
cc100_words, cc100_X = load_words_X(CC / "cc_pt_top100k.npz")
wiki100_words, wiki100_X = load_words_X(CACHE / "muse_pt_top100k.npz")
cc100_idx = {w: i for i, w in enumerate(cc100_words)}
wiki100_idx = {w: i for i, w in enumerate(wiki100_words)}

muse_pt_idx = {w: i for i, w in enumerate(pt_words)}
common = [w for w in cc_pt_words if w in muse_pt_idx]
rng9 = np.random.default_rng(SEED)
perm = rng9.permutation(len(common))
held = [common[i] for i in perm[:1000]]
train_w = [common[i] for i in perm[1000:]]
print(f"vocab comum cc20k ∩ muse-pt: {len(common)} "
      f"(treino={len(train_w)}, held-out={len(held)})")

cc20_idx = {w: i for i, w in enumerate(cc_pt_words)}
A_tr = cc_pt_X[:, [cc20_idx[w] for w in train_w]]
B_tr = pt_X[:, [muse_pt_idx[w] for w in train_w]]
U, _, Vt = np.linalg.svd(B_tr @ A_tr.T)
W = U @ Vt

A_ho = cc_pt_X[:, [cc20_idx[w] for w in held]]
B_ho = pt_X[:, [muse_pt_idx[w] for w in held]]
WA = W @ A_ho
cos_ho = np.sum(WA * B_ho, axis=0) / (
    np.linalg.norm(WA, axis=0) * np.linalg.norm(B_ho, axis=0))
proc_val = float(cos_ho.mean())
print(f"Procrustes: cosseno médio held-out = {proc_val:.3f}")

# %% [markdown]
# **(b)** Pares de diminutivos: offsets `v(diminutivo) − v(base)` calculados
# **no espaço cc** (top-100k, sem projetar) e coerência interna (cosseno
# médio par a par dos offsets). Comparação: a mesma lista nos vetores
# Wikipédia (muse_pt_top100k) e um baseline de offsets de pares aleatórios.

# %%
DIM_PAIRS = [("casa", "casinha"), ("gato", "gatinho"), ("pouco", "pouquinho"),
             ("amigo", "amiguinho"), ("café", "cafezinho"),
             ("beijo", "beijinho"), ("pedaço", "pedacinho"),
             ("cidade", "cidadezinha"), ("irmã", "irmãzinha"),
             ("cavalo", "cavalinho"), ("bola", "bolinha"),
             ("carro", "carrinho"), ("barco", "barquinho"),
             ("igreja", "igrejinha"), ("fazenda", "fazendinha"),
             ("escola", "escolinha"), ("praça", "pracinha"),
             ("neto", "netinho"), ("irmão", "irmãozinho"),
             ("velho", "velhinho"), ("baixo", "baixinho"),
             ("bonito", "bonitinho"), ("filho", "filhinho"),
             ("mãe", "mãezinha"), ("pai", "paizinho"), ("hora", "horinha"),
             ("jeito", "jeitinho"), ("coisa", "coisinha"),
             ("festa", "festinha"), ("presente", "presentinho")]


def offsets_in(pairs, X, idx):
    found, offs = [], []
    for base, dim in pairs:
        if base in idx and dim in idx:
            found.append((base, dim))
            offs.append(X[:, idx[dim]] - X[:, idx[base]])
    return found, np.array(offs).T  # (dim x n)


def mean_pairwise_cos(O):
    On = O / np.maximum(np.linalg.norm(O, axis=0), 1e-12)
    G = On.T @ On
    n = G.shape[0]
    return float((G.sum() - n) / (n * (n - 1)))


found_cc, off_cc = offsets_in(DIM_PAIRS, cc100_X, cc100_idx)
found_wk, off_wk = offsets_in(DIM_PAIRS, wiki100_X, wiki100_idx)
coh_cc9 = mean_pairwise_cos(off_cc)
coh_wk9 = mean_pairwise_cos(off_wk)

# baseline: offsets de pares aleatórios do vocabulário cc (mesmo n)
alpha_ws = [w for w in cc100_words[:20000] if w.isalpha() and len(w) > 2]
rb = np.random.default_rng(SEED)
rand_pairs = [(alpha_ws[i], alpha_ws[j]) for i, j in
              rb.choice(len(alpha_ws), (len(found_cc), 2), replace=False)]
_, off_rand = offsets_in(rand_pairs, cc100_X, cc100_idx)
coh_rand = mean_pairwise_cos(off_rand)

print(f"pares no cc100k: {len(found_cc)}/{len(DIM_PAIRS)}; "
      f"no wiki100k: {len(found_wk)}/{len(DIM_PAIRS)}")
print(f"coerência interna: cc={coh_cc9:.3f} wiki={coh_wk9:.3f} "
      f"baseline aleatório={coh_rand:.3f}")

# %% [markdown]
# **(c)** Projetamos os offsets cc para o espaço MUSE via `W` e medimos θ
# no frame padrão (`Frame().theta`). Referências: θ médio dos offsets
# wiki (~43.4°) e das palavras PT comuns (~41.6°). Reportamos como sair.

# %%
theta_cc = f.theta(W @ off_cc)
theta_wk = f.theta(off_wk)
# referência: θ de palavras PT comuns (200 palavras dos pares de treino)
ref_ws = [p for _, p in f.pairs[:200]]
theta_ref = f.theta(f.pt_X[:, [f.pt_idx[w] for w in ref_ws]])
print(f"θ médio: diminutivos cc-projetados={theta_cc.mean():.2f}° "
      f"wiki={theta_wk.mean():.2f}° palavras PT comuns={theta_ref.mean():.2f}°")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(theta_cc, bins=15, alpha=0.6, label="offsets cc (projetados)")
ax.hist(theta_wk, bins=15, alpha=0.6, label="offsets wiki")
ax.axvline(45, color="gray", ls=":")
ax.set_xlabel("θ (graus)")
ax.set_ylabel("nº de offsets")
ax.set_title("θ dos offsets de diminutivo")
ax.legend()
fig.tight_layout()

exp9_out = {
    "procrustes": {"n_vocab_comum": len(common), "n_treino": len(train_w),
                   "n_heldout": len(held),
                   "cos_medio_heldout": round(proc_val, 4)},
    "n_pares_cc100k": len(found_cc), "n_pares_wiki100k": len(found_wk),
    "coerencia_offsets": {"cc": round(coh_cc9, 4), "wiki": round(coh_wk9, 4),
                          "baseline_aleatorio": round(coh_rand, 4)},
    "theta_medio": {"diminutivos_cc_projetados": round(float(theta_cc.mean()), 2),
                    "diminutivos_wiki": round(float(theta_wk.mean()), 2),
                    "palavras_pt_comuns": round(float(theta_ref.mean()), 2)},
    "theta_cc_por_par": {f"{b}->{d}": round(float(t), 2)
                         for (b, d), t in zip(found_cc, theta_cc)},
}
with open(RESULTS / "exp9.json", "w", encoding="utf-8") as fh:
    json.dump(exp9_out, fh, ensure_ascii=False, indent=2)
print("salvo:", RESULTS / "exp9.json")

# %% [markdown]
# **Leitura**: o Exp. 4 testa se a banda compartilhada acompanha a
# proximidade tipológica; o Exp. 6 mostra o que acontece quando não há
# estrutura compartilhada nenhuma (controle negativo); o Exp. 9 verifica
# se a morfologia produtiva do diminutivo emerge como direção linear em
# texto informal. Os números salvos em `paper_results/exp{4,6,9}.json`
# valem como saíram, confirmem ou não as hipóteses.
