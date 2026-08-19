"""Generic plotting layer (matplotlib).

Nothing here knows about MNIST: every "draw one sample vector" operation goes
through a *renderer*, a callable ``renderer(ax, vector, title=None)``. Use
:class:`ImageRenderer` for image data (any shape), :class:`LineRenderer` for
embeddings / spectra / time series, or pass your own callable.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from .angles import get_nonzero_per_column


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class ImageRenderer:
    """Render a flattened image vector as a heatmap. ``shape`` = (h, w)."""

    def __init__(self, shape=(28, 28), cmap="viridis", transpose=False):
        self.shape = shape
        self.cmap = cmap
        # transpose=True for column-major-flattened vectors (Julia's vec);
        # the datasets in gsvdlib flatten row-major, so the default is False.
        self.transpose = transpose

    def __call__(self, ax, vector, title=None):
        img = np.asarray(vector).reshape(self.shape)
        if self.transpose:
            img = img.T
        ax.imshow(img, cmap=self.cmap)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=9)


class LineRenderer:
    """Render a vector as a curve (embeddings, spectra, series)."""

    def __init__(self, **plot_kwargs):
        self.plot_kwargs = plot_kwargs

    def __call__(self, ax, vector, title=None):
        ax.plot(np.asarray(vector).ravel(), **self.plot_kwargs)
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title, fontsize=9)


class BarRenderer:
    """Render a vector as bars (sparse / low-dimensional vectors)."""

    def __init__(self, **bar_kwargs):
        self.bar_kwargs = bar_kwargs

    def __call__(self, ax, vector, title=None):
        v = np.asarray(vector).ravel()
        ax.bar(np.arange(v.size), v, **self.bar_kwargs)
        if title:
            ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Sample grids and reconstructions
# ---------------------------------------------------------------------------

def plot_samples(M, renderer, k=None, mean=None, title=None,
                 max_per_row=10, save=None, col_range=None):
    """Grid of the columns of ``M`` drawn with ``renderer``.

    ``mean`` (a d-vector) is added back to every column if given;
    ``col_range = (start, stop)`` restricts to a column slice (0-based,
    stop exclusive); ``k`` caps the number of samples shown.
    """
    M = np.atleast_2d(np.asarray(M, dtype=float))
    if col_range is not None:
        M = M[:, col_range[0]:col_range[1]]
    if mean is not None:
        M = M + np.asarray(mean, dtype=float).reshape(-1, 1)
    n = M.shape[1] if k is None else min(k, M.shape[1])

    ncols = min(n, max_per_row)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.4 * ncols, 1.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        renderer(axes[i], M[:, i])
    for ax in axes[n:]:
        ax.set_axis_off()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_vec(x, renderer, title=None, save=None):
    """Draw a single vector."""
    fig, ax = plt.subplots(figsize=(3, 3))
    renderer(ax, x, title=title)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_recon(D, H, i, mean, renderer, title=None):
    """Reconstruction of component ``i``: ``mean + D[i, j] * H[:, i]`` where
    ``j`` is the first nonzero of row ``i`` (the Julia ``plot_recon_A/B``,
    which were identical). ``H`` here is oriented so components are columns
    (pass ``H.T`` if your H has components as rows)."""
    D = np.asarray(D)
    H = np.asarray(H)
    row = D[i, :]
    nz = np.flatnonzero(row)
    coeff = row[nz[0]] if nz.size else 0.0
    vec = np.asarray(mean, dtype=float).ravel() + coeff * H[:, i]
    return plot_vec(vec, renderer, title=title or f"Reconstruction, component {i}")


def save_component_images(H, indices, renderer, prefix="H_component"):
    """Save one figure per requested row of ``H`` (components as rows)."""
    paths = []
    for idx in indices:
        fig = plot_vec(np.asarray(H)[idx, :], renderer, title=f"H component {idx}")
        path = f"{prefix}_{idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def animate_components(M, renderer, save="components.gif", fps=15,
                       mean=None, col_range=None, title="Component"):
    """GIF scrolling through the columns of ``M`` (the Julia ``gif_matrix``)."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    start = 0 if col_range is None else col_range[0]
    if col_range is not None:
        M = M[:, col_range[0]:col_range[1]]
    if mean is not None:
        M = M + np.asarray(mean, dtype=float).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(3, 3.2))

    def draw(i):
        ax.clear()
        renderer(ax, M[:, i], title=f"{title} {start + i + 1} / {start + M.shape[1]}")

    anim = animation.FuncAnimation(fig, draw, frames=M.shape[1])
    anim.save(save, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return save


# ---------------------------------------------------------------------------
# Angle plots
# ---------------------------------------------------------------------------

def plot_angles(D, target_deg=45.0, use="cos"):
    """Per-column angles of a blocked factor with a reference line."""
    vals = np.clip(get_nonzero_per_column(D), -1.0, 1.0)
    angles = np.degrees(np.arccos(vals) if use == "cos" else np.arcsin(vals))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles, marker="o", ms=3)
    ax.axhline(target_deg, ls="--", color="red")
    ax.set_xlabel("Index")
    ax.set_ylabel("Angle (degrees)")
    ax.grid(True, alpha=0.3)
    return fig


def plot_angle_histogram(angles_A, angles_B, name_A, name_B, bins=30,
                         alpha=0.6, xlims=(0, 90), ylims=None,
                         colors=("steelblue", "coral"), save=None,
                         figsize=(12, 5)):
    """Overlaid theta histograms for the two classes, with mean lines
    (generic version of the MNIST/Fashion histogram functions)."""
    angles_A = np.asarray(angles_A)
    angles_B = np.asarray(angles_B)
    fig, ax = plt.subplots(figsize=figsize)
    edges = np.linspace(*xlims, bins + 1)
    for angles, name, color in ((angles_A, name_A, colors[0]),
                                (angles_B, name_B, colors[1])):
        ax.hist(angles, bins=edges, alpha=alpha, color=color,
                label=f"{name} (σ={np.std(angles, ddof=1):.1f}°)")
        ax.axvline(np.mean(angles), color=color, ls="--", lw=3,
                   label=f"Mean {name}: {np.mean(angles):.1f}°")
    ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)
    ax.set_xlabel("Angle θ (degrees)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def plot_posterior(angles_A, angles_B, name_A, name_B, bins=30,
                   xlims=(0, 90), colors=("steelblue", "coral"),
                   save=None, figsize=(12, 5)):
    """Empirical P(class | theta) per bin as a stacked bar chart; empty bins
    get a translucent background hinting the side of 45 degrees."""
    angles_A = np.asarray(angles_A)
    angles_B = np.asarray(angles_B)
    edges = np.linspace(*xlims, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    counts_A, _ = np.histogram(angles_A, bins=edges)
    counts_B, _ = np.histogram(angles_B, bins=edges)
    total = counts_A + counts_B
    empty = total == 0
    total_safe = np.where(empty, 1, total)
    post_A = counts_A / total_safe
    post_B = counts_B / total_safe

    fig, ax = plt.subplots(figsize=figsize)
    mid = (xlims[0] + xlims[1]) / 2
    for i in np.flatnonzero(empty):
        color = colors[0] if centers[i] < mid else colors[1]
        ax.bar(centers[i], 1.0, width=width, color=color, alpha=0.3)
    ax.bar(centers, post_A + post_B, width=width, color=colors[1], label=name_B)
    ax.bar(centers, post_A, width=width, color=colors[0], label=name_A)
    ax.set_xlim(*xlims)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Angle θ (degrees)")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig
