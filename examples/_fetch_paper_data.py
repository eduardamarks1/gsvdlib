# -*- coding: utf-8 -*-
"""Baixa e cacheia dados para experimentos de embeddings multilingues.

1. Vetores MUSE alinhados top-20k (es, de, zh)
2. Dicionarios MUSE (pt-es, en-de, en-zh, en-es), com fallback invertido
3. fastText CommonCrawl nao-alinhado (cc.pt, cc.en) via streaming gzip
"""
import gzip
import io
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

MUSE_CACHE = Path.home() / ".gsvdlib" / "muse"
CC_CACHE = Path.home() / ".gsvdlib" / "cc"
MUSE_CACHE.mkdir(parents=True, exist_ok=True)
CC_CACHE.mkdir(parents=True, exist_ok=True)

TOP_K = 20000
UA = {"User-Agent": "gsvdlib"}


def stream_top_vectors(reader, top_k):
    header = reader.readline()  # "n_words dim"
    dim = int(header.split()[1])
    words, vecs = [], []
    for line in reader:
        parts = line.rstrip().split(" ")
        if len(parts) != dim + 1:
            continue
        words.append(parts[0])
        vecs.append(np.asarray(parts[1:], dtype=np.float32))
        if len(words) >= top_k:
            break
    return words, np.stack(vecs, axis=1)  # (dim, n)


def fetch_muse_vectors():
    for lang in ("es", "de", "zh"):
        out = MUSE_CACHE / f"wiki_multi_{lang}_top20k.npz"
        if out.exists():
            print("cache ja existe:", out)
            continue
        url = f"https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.{lang}.vec"
        print(f"baixando top {TOP_K} vetores {lang} (streaming)...")
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as resp:
                reader = io.TextIOWrapper(resp, encoding="utf-8", errors="ignore")
                words, X = stream_top_vectors(reader, TOP_K)
        except Exception as e:
            print(f"  FALHA {lang}: {e!r} -- seguindo sem {lang}")
            continue
        np.savez_compressed(out, words=np.array(words), X=X)
        print(f"  salvo: {out}  X{X.shape}")


def fetch_dicts():
    for pair in ("pt-es", "en-de", "en-zh", "en-es"):
        out = MUSE_CACHE / f"dict_{pair}.txt"
        if out.exists():
            print("cache ja existe:", out)
            continue
        a, b = pair.split("-")
        for cand in (pair, f"{b}-{a}"):
            url = f"https://dl.fbaipublicfiles.com/arrival/dictionaries/{cand}.txt"
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    text = resp.read().decode("utf-8")
                out.write_text(text, encoding="utf-8")
                nlines = text.count("\n")
                print(f"  dict {pair}: baixado como {cand} ({nlines} linhas) -> {out}")
                break
            except urllib.error.HTTPError as e:
                print(f"  dict {cand}: HTTP {e.code}")
        else:
            print(f"  FALHA: nenhum dicionario para {pair}")


def fetch_cc(lang, top_k, out_name):
    out = CC_CACHE / out_name
    if out.exists():
        print("cache ja existe:", out)
        return
    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz"
    print(f"baixando top {top_k} vetores cc.{lang} (streaming gzip)...")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as resp:
        gz = gzip.GzipFile(fileobj=resp)
        reader = io.TextIOWrapper(gz, encoding="utf-8", errors="ignore")
        words, X = stream_top_vectors(reader, top_k)
    np.savez_compressed(out, words=np.array(words), X=X)
    print(f"  salvo: {out}  X{X.shape}")


def main():
    fetch_muse_vectors()
    fetch_dicts()
    fetch_cc("pt", TOP_K, "cc_pt_top20k.npz")
    fetch_cc("en", TOP_K, "cc_en_top20k.npz")
    fetch_cc("pt", 100000, "cc_pt_top100k.npz")


if __name__ == "__main__":
    main()
