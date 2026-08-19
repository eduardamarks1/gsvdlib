# -*- coding: utf-8 -*-
"""Setup compartilhado dos experimentos do paper_explorations.ipynb.

Carrega o cache MUSE pt/en, monta o vocabulario pareado, o frame GSVD
(identico ao demo_word_embeddings.ipynb) e helpers usados por varios
experimentos. Importado tanto pelos scripts de teste quanto pelo notebook.
"""
from pathlib import Path

import numpy as np

from gsvdlib import gsvd, theta_angles, get_angles_C, get_most_similar_row_H

CACHE = Path.home() / ".gsvdlib" / "muse"
RESULTS = Path(__file__).parent / "paper_results"
RESULTS.mkdir(exist_ok=True)


def load_muse_pair(npz_name="muse_pt_en_top20000.npz"):
    """Vocabularios e vetores (dim x n) do cache MUSE pt/en."""
    d = np.load(CACHE / npz_name)
    pt_words = [str(w) for w in d["pt_words"]]
    en_words = [str(w) for w in d["en_words"]]
    return (pt_words, d["pt_X"].astype(float),
            en_words, d["en_X"].astype(float),
            [str(w) for w in d["dict_en"]], [str(w) for w in d["dict_pt"]])


def build_pairs(pt_words, en_words, dict_en, dict_pt):
    """Pares de traducao usaveis (mesma filtragem do demo)."""
    pt_idx = {w: i for i, w in enumerate(pt_words)}
    en_idx = {w: i for i, w in enumerate(en_words)}
    pairs, seen_en, seen_pt = [], set(), set()
    for en_w, pt_w in zip(dict_en, dict_pt):
        if (en_w in en_idx and pt_w in pt_idx
                and en_w.isalpha() and pt_w.isalpha()
                and len(en_w) > 2 and len(pt_w) > 2
                and en_w != pt_w
                and en_w not in seen_en and pt_w not in seen_pt):
            pairs.append((en_w, pt_w))
            seen_en.add(en_w)
            seen_pt.add(pt_w)
    return pairs, pt_idx, en_idx


def build_frame(A_feat, B_feat, n_A=900, n_B=800, seed=1234):
    """Frame GSVD ordenado a partir de duas matrizes (features x amostras).

    Centra cada lado, decompoe as transpostas (formulacao co-span, como no
    paper) e devolve (g, A_centrada, B_centrada). Generico: serve para
    qualquer par de linguas ou de espacos.
    """
    rng = np.random.default_rng(seed)
    A = A_feat[:, rng.permutation(A_feat.shape[1])[:n_A]]
    B = B_feat[:, rng.permutation(B_feat.shape[1])[:n_B]]
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    g = gsvd(A.T, B.T).sorted()
    return g, A, B


class Frame:
    """Empacota o frame pt/en padrao + helpers dependentes dele."""

    def __init__(self, seed=1234):
        (self.pt_words, self.pt_X, self.en_words, self.en_X,
         dict_en, dict_pt) = load_muse_pair()
        self.pairs, self.pt_idx, self.en_idx = build_pairs(
            self.pt_words, self.en_words, dict_en, dict_pt)

        # dicionario de traducao (conjuntos, para checar coerencia)
        self.en2pt, self.pt2en = {}, {}
        for e, p in zip(dict_en, dict_pt):
            self.en2pt.setdefault(e, set()).add(p)
            self.pt2en.setdefault(p, set()).add(e)

        # amostra as palavras do frame a partir dos pares (como no demo)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.pairs))
        train = order[:1200]
        pt_sel = [self.pairs[i][1] for i in train]
        en_sel = [self.pairs[i][0] for i in train]
        A_all = self.pt_X[:, [self.pt_idx[w] for w in pt_sel]]
        B_all = self.en_X[:, [self.en_idx[w] for w in en_sel]]
        self.g, self.A, self.B = build_frame(A_all, B_all, seed=seed)
        self.angles = get_angles_C(self.g.C)         # por direcao, 0-90
        self.Hinv_T = np.linalg.inv(self.g.H.T)      # H e 300x300 invertivel
        self.test_pairs = [self.pairs[i] for i in order[1200:2200]]

    # --- helpers dependentes do frame -----------------------------------
    def v_pt(self, w):
        return self.pt_X[:, self.pt_idx[w]]

    def v_en(self, w):
        return self.en_X[:, self.en_idx[w]]

    def coords(self, V):
        """Coordenadas no frame H (V: dim x n) -> (300 x n)."""
        return self.Hinv_T @ np.atleast_2d(V.T).T

    def band_mask(self, lo, hi):
        return (self.angles >= lo) & (self.angles <= hi)

    def band_project(self, V, mask):
        """Reconstroi V mantendo so as direcoes do mask (band-pass)."""
        c = self.coords(V)
        c = c * mask[:, None]
        return self.g.H.T @ c

    def energy_profile(self, v):
        c = (self.Hinv_T @ v.reshape(-1, 1)).ravel() ** 2
        return c / c.sum()

    def theta(self, V):
        return theta_angles(np.atleast_2d(V.T).T, self.g.C, self.g.S, self.g.H)

    def top_words(self, direction, k=10):
        sp = get_most_similar_row_H(direction, self.pt_X.T)
        se = get_most_similar_row_H(direction, self.en_X.T)
        tp = [self.pt_words[j] for j in np.argsort(-np.abs(sp))[:k]]
        te = [self.en_words[j] for j in np.argsort(-np.abs(se))[:k]]
        return tp, te


def nn_search(query, X, exclude=None, k=5):
    """Vizinhos por cosseno: query (dim,), X (dim x n). Retorna indices."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=0), 1e-12)
    q = query / max(np.linalg.norm(query), 1e-12)
    sims = q @ Xn
    if exclude:
        sims[list(exclude)] = -np.inf
    return np.argsort(-sims)[:k]
