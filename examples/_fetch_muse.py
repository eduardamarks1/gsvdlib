# -*- coding: utf-8 -*-
"""Baixa (streaming, so o topo dos arquivos) os embeddings MUSE alinhados
pt/en e o dicionario bilingue en-pt, e salva um cache .npz local."""
import io
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".gsvdlib" / "muse"
CACHE.mkdir(parents=True, exist_ok=True)
TOP_K = 20000

VEC_URLS = {
    "pt": "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.pt.vec",
    "en": "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec",
}
DICT_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-pt.txt"


def stream_top_vectors(url, top_k):
    req = urllib.request.Request(url, headers={"User-Agent": "gsvdlib"})
    words, vecs = [], []
    with urllib.request.urlopen(req, timeout=120) as resp:
        reader = io.TextIOWrapper(resp, encoding="utf-8", errors="ignore")
        header = reader.readline()  # "n_words dim"
        dim = int(header.split()[1])
        for line in reader:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            words.append(parts[0])
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
            if len(words) >= top_k:
                break
    return words, np.stack(vecs, axis=1)  # (dim, n)


def main():
    out = CACHE / f"muse_pt_en_top{TOP_K}.npz"
    if out.exists():
        print("cache ja existe:", out)
        return

    print("baixando dicionario en-pt...")
    req = urllib.request.Request(DICT_URL, headers={"User-Agent": "gsvdlib"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        pairs = [l.split() for l in resp.read().decode("utf-8").splitlines()]
    pairs = [(en, pt) for en, pt, *rest in (p for p in pairs if len(p) >= 2)]
    print(f"{len(pairs)} pares en-pt")

    for lang, url in VEC_URLS.items():
        print(f"baixando top {TOP_K} vetores {lang} (streaming)...")
        words, X = stream_top_vectors(url, TOP_K)
        np.savez_compressed(CACHE / f"tmp_{lang}.npz", words=np.array(words), X=X)
        print(f"  {lang}: {X.shape}")

    pt = np.load(CACHE / "tmp_pt.npz", allow_pickle=False)
    en = np.load(CACHE / "tmp_en.npz", allow_pickle=False)
    np.savez_compressed(
        out,
        pt_words=pt["words"], pt_X=pt["X"],
        en_words=en["words"], en_X=en["X"],
        dict_en=np.array([p[0] for p in pairs]),
        dict_pt=np.array([p[1] for p in pairs]),
    )
    (CACHE / "tmp_pt.npz").unlink()
    (CACHE / "tmp_en.npz").unlink()
    print("salvo:", out)


if __name__ == "__main__":
    main()
