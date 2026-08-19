"""Generic dataset layer.

Everything downstream (sampling, GSVD base, classification, plots) works on
"vectors as columns" and never knows where they came from. Any data source is
a :class:`VectorDataset`:

- :class:`ArrayDataset` wraps a plain matrix + labels (word embeddings,
  activations, tabular features, anything).
- :class:`MNISTDataset` / :class:`FashionMNISTDataset` download the IDX files
  once into ``data_dir`` and expose the images as flattened columns in [0, 1].
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorDataset(Protocol):
    def get_class(self, label, split: str = "train") -> np.ndarray:
        """All samples of ``label`` as columns of a (d, n) array."""
        ...

    def class_name(self, label) -> str:
        """Human-readable name for a label (used in plot legends)."""
        ...


class ArrayDataset:
    """Wrap an in-memory matrix (samples as columns) with per-column labels.

    This is the fully generic case: e.g. word embeddings with
    ``X`` = (embedding_dim, n_words) and ``labels`` = word categories.
    ``split`` is ignored unless a dict of splits is provided.
    """

    def __init__(self, X=None, labels=None, splits=None, names=None):
        if splits is None:
            if X is None or labels is None:
                raise ValueError("provide either (X, labels) or splits")
            splits = {"train": (X, labels), "test": (X, labels)}
        self._splits = {
            k: (np.asarray(v[0], dtype=float), np.asarray(v[1]))
            for k, v in splits.items()
        }
        self._names = dict(names) if names else {}

    def get_class(self, label, split="train"):
        X, labels = self._splits[split]
        return X[:, labels == label]

    def class_name(self, label):
        return str(self._names.get(label, label))


# ---------------------------------------------------------------------------
# MNIST / Fashion-MNIST (IDX format)
# ---------------------------------------------------------------------------

def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        zero, dtype_code, ndim = struct.unpack(">HBB", f.read(4))
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


class _IdxImageDataset:
    """Shared implementation for the two MNIST variants."""

    BASE_URL: str = ""
    FILES = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    }
    NAMES: dict = {}

    def __init__(self, data_dir: Optional[str] = None):
        default = Path.home() / ".gsvdlib" / self.__class__.__name__.lower()
        self.data_dir = Path(data_dir) if data_dir else default
        self._cache = {}

    def _load(self, split):
        if split not in self._cache:
            img_name, lbl_name = self.FILES[split]
            self.data_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for name in (img_name, lbl_name):
                path = self.data_dir / name
                if not path.exists():
                    urllib.request.urlretrieve(self.BASE_URL + name, path)
                paths.append(path)
            imgs = _read_idx(paths[0]).astype(float) / 255.0   # (n, 28, 28)
            labels = _read_idx(paths[1]).astype(int)
            X = imgs.reshape(imgs.shape[0], -1).T              # (784, n)
            self._cache[split] = (X, labels)
        return self._cache[split]

    def get_class(self, label, split="train"):
        X, labels = self._load(split)
        return X[:, labels == label]

    def class_name(self, label):
        return str(self.NAMES.get(label, label))

    @property
    def sample_shape(self):
        return (28, 28)


class MNISTDataset(_IdxImageDataset):
    BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    NAMES = {i: f"Digit {i}" for i in range(10)}


class FashionMNISTDataset(_IdxImageDataset):
    BASE_URL = ("https://raw.githubusercontent.com/zalandoresearch/"
                "fashion-mnist/master/data/fashion/")
    NAMES = {
        0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
    }


# ---------------------------------------------------------------------------
# Generic sampling / preprocessing helpers
# ---------------------------------------------------------------------------

def sample_pair(ds: VectorDataset, label_A, label_B, n_A, n_B,
                split="train", seed=1234):
    """Sample ``n_A`` columns of ``label_A`` and ``n_B`` of ``label_B``.

    Pass ``n_A == n_B`` for a balanced pair. Returns ``(A, B)``.
    """
    rng = np.random.default_rng(seed)
    out = []
    for label, n in ((label_A, n_A), (label_B, n_B)):
        X = ds.get_class(label, split=split)
        if X.shape[1] < n:
            raise ValueError(
                f"Not enough samples of {label!r} in split={split!r} "
                f"(need {n}, have {X.shape[1]})")
        idx = rng.permutation(X.shape[1])[:n]
        out.append(X[:, idx])
    return tuple(out)


def balanced_count(ds: VectorDataset, label_A, label_B, split="test"):
    """Largest balanced per-class count available for the pair."""
    return min(ds.get_class(label_A, split=split).shape[1],
               ds.get_class(label_B, split=split).shape[1])


def center(X):
    """Feature-center columns. Returns ``(X_centered, mean_vector)``."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=1)
    return X - mean[:, None], mean
