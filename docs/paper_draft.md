# The θ-Spectrum of the GSVD Orders Embedding Directions by Language Specificity

*Draft v0.1 — 2026-08-19. Companion code: `gsvdlib` (Python) and
`examples/paper_explorations.ipynb`; numerical results in
`examples/paper_results/*.json`.*

## Abstract

The Generalized Singular Value Decomposition (GSVD) of a pair of datasets
yields a joint frame in which every direction carries an alignment angle
θ ∈ [0°, 90°] measuring whether it is better explained by dataset A, dataset
B, or shared by both. Prior work used this frame to compare image classes
(MNIST digit pairs), where θ acts as an interpretable per-sample diagnostic.
Here we transplant the same algebra to cross-lingual word embeddings
(MUSE-aligned fastText, Portuguese × English) and show that the θ-spectrum is
not merely a similarity summary: it **orders the directions of the shared
space by language specificity**, and that ordering is functional. Translation
coherence of direction-level word neighborhoods peaks exactly in the shared
band (0.23 at θ≈47° vs 0.15 at the extremes); translation retrieval using
only the 72 most central directions (24% of the space) outperforms
size-matched random subspaces at every intermediate budget; cross-lingual
analogy offsets transfer with accuracies that track the semantic type of the
relation (gender 99%, capital–country 71%, plural 28%); and a non-aligned
control collapses every one of these signals (coherence 0.00, P@1 0.007).
Applied layer-wise to mBERT, the spectrum reproduces the known
inverted-U interlinguality profile with layer 7 at the peak. Unlike CKA or
retrieval scores, which output one number per dataset pair, the GSVD frame
outputs one interpretable number **per direction**, with exact linear-algebra
semantics inherited from the decomposition. We also report three carefully
controlled negative results that sharpen what the spectrum does *not* encode.

## 1. Introduction

Dataset-comparison methods in representation learning typically produce a
single scalar per pair of datasets (CKA, SVCCA scores, retrieval precision).
Such summaries answer *how much* two representations agree but not *where* —
which directions of the space carry the agreement, and which carry
dataset-specific structure.

Previous work [the MNIST paper] introduced a comparison frame built from the
GSVD of two data matrices A and B sharing an ambient space. Writing
A = U·C·H and B = Ṽ·S·H with CᵀC + SᵀS = I, every column of the joint factor
H receives an angle θ = atan2(s, c) from the paired cosine/sine values. On
MNIST digit pairs this produced: (i) a per-sample classifier from the angle
θ(z) of a test vector, reaching 90–97% on digit pairs with low CKA and
degrading gracefully as CKA grows; and (ii) interpretable extreme and shared
directions (the "principal shared component" at θ≈45°).

This paper asks what the same frame reveals when the two datasets are the
*same vocabulary in two languages*: Portuguese and English word vectors in a
MUSE-aligned multilingual space. Our claim:

> **The θ-spectrum orders the directions of an aligned multilingual space by
> how language-bound they are, and this ordering predicts which linguistic
> functions each region of the space supports.**

Contributions:

1. **Direction-level interpretability with a quantitative label.** We
   introduce *translation coherence* per direction and show it peaks in the
   shared band (§4.2) — turning the qualitative "spectrum reading" into a
   measurable curve.
2. **Functional band analysis.** Band-pass projections of the space show
   that translation retrieval concentrates in the central band: 100 central
   directions retain 84% of full-space P@1, beating random subspaces at all
   intermediate budgets (§4.3).
3. **Relation transfer.** Cross-lingual analogy accuracy stratifies by
   relation type in the same order as the cross-language invariance of the
   offset vectors (§4.4).
4. **Generalization and controls.** The spectrum's shared band tracks
   language proximity across four language pairs (§4.5); it collapses
   entirely on non-aligned embeddings (§4.6); and applied per-layer to mBERT
   it recovers the inverted-U interlinguality curve with layer 7 at the peak
   (§4.7).
5. **Negative results, reported.** Shared-band energy of an offset does not
   predict its analogy accuracy; entity words do not concentrate extreme
   energy once frequency is matched; and the Portuguese diminutive is not a
   clean linear direction even in informal text (§5).

## 2. Method

### 2.1 The GSVD frame and the alignment angle

Given A ∈ ℝ^{n_A×d} and B ∈ ℝ^{n_B×d} (samples × features; here d = 300 and
rows are word vectors), the GSVD produces

  A = U·C·H,  B = Ṽ·S·H,  CᵀC + SᵀS = I,

where H ∈ ℝ^{t×d} spans the joint row space, and the diagonal blocks of C
and S pair a cosine c_i and sine s_i to each direction H_i, giving
θ_i = atan2(s_i, c_i). Directions are sorted so θ increases; blocks with
c_i = 1 (θ = 0°) are A-exclusive and s_i = 1 (θ = 90°) B-exclusive. For a
test vector v, solving Hᵀx = v and comparing ‖C·x‖ with ‖S·x‖ yields the
per-sample angle θ(v). (Implementation: `gsvdlib`, a NumPy/SciPy
reimplementation of the LAPACK-backed Julia pipeline via the Paige–Saunders
QR + CS construction, validated to machine precision against the original.)

### 2.2 From images to embeddings

On MNIST, A and B are two digit classes and the ambient space is pixels; the
useful signal was the per-sample classifier θ(z). On embeddings, A holds
MUSE vectors of 900 Portuguese words and B of 800 English words (from 8,331
dictionary-paired types in the top-20k vocabularies, mean-centered per side).
Both matrices are full-rank in d = 300, so the entire space is intersection
(bl = br = 0, r = 300): *all* structure lives in the angles, the regime where
a block count or a scalar score is least informative. The observed spectrum
spans θ ∈ [15.1°, 70.3°].

Two operators used throughout: the **energy profile** of a vector
(e_i = x_i²/‖x‖² for coordinates x in the H-frame) and the **band-pass
projection** (reconstruction keeping only directions with θ in a band).

### 2.3 Data and protocol

MUSE-aligned fastText (wiki.multi.*, 300d, top-20k per language); MUSE
bilingual dictionaries for pairing and for coherence scoring; fastText
CommonCrawl (cc.pt, cc.en) as the non-aligned control and informal-text
source; bert-base-multilingual-cased for the layer-wise study. 1,200 word
pairs build each frame; 1,000 held-out pairs are used for evaluation. Full
protocols in `paper_explorations.ipynb`.

## 3. Recap: what the frame does on MNIST

For calibration we summarize the image results [from the original paper's
code, reproduced by `gsvdlib`]: on digit pairs (1,5), (0,7), (3,9), (4,9)
the θ(z) classifier attains 96.1%, 97.2%, 96.0%, 89.9% accuracy, with CKA
0.11, 0.22, 0.34, 0.74 respectively — accuracy falls as the pair's geometry
overlaps. On CIFAR-10 grayscale pairs (CKA 0.92–0.98) the distributions
collapse onto 45–46° and classification is weak (51–59%), which is the
diagnostic working as designed: raw-pixel geometry there is almost entirely
shared. Aligned embeddings (CKA 0.85) sit near this saturated regime — which
is precisely why the *per-direction* analysis of this paper, rather than the
per-sample classifier, is the right tool for them.

## 4. Results

### 4.1 The spectrum reads as a gradient of language specificity

Walking H from θ = 15° to 70° and listing the nearest vocabulary words on
each side: Portuguese proper names (15°) → Portuguese-inflected vocabulary
(20–25°) → a wide bilingual band where the PT and EN neighbor lists are
literal translations (*desenvolvimento/development* at 30°; *biologia,
vertebrados / biology, vertebrates* at 40°; *imparcial/impartial* at 45°) →
English proper names (70°). The frame does not separate topics; it separates
*how language-bound* each direction is.

### 4.2 Translation coherence peaks in the shared band (Exp. 7)

For each of the 300 directions we take the top-10 aligned words per language
and score the symmetric fraction that are dictionary translations of each
other. Binned by θ (5° bins), coherence forms a hat curve peaking at
**0.231 in the 45–50° bin**, vs 0.146 below 30° and 0.151 above 60°
(corr(coherence, −|θ−45°|) = 0.22). High-coherence directions are
near-parallel bilingual word lists (e.g., direction 94, θ = 35.8°, coherence
0.70); zero-coherence directions are monolingual name/fauna clusters
(direction 298, θ = 69.1°). This quantifies §4.1: **traduzibility is a
function of θ.**

### 4.3 The central band carries translation (Exps. 2–3)

Band-pass the space and test what each region supports:

| band (dirs) | translation P@1 | P@5 | analogy top-1 |
|---|---|---|---|
| PT-side, θ<40° (123) | 0.650 | 0.782 | 0.510 |
| **shared, 40–50° (72)** | **0.672** | 0.780 | 0.435 |
| EN-side, θ>50° (105) | 0.576 | 0.720 | 0.585 |
| random control (72) | 0.558 | 0.712 | 0.370 |

The shared band achieves the best P@1 with the fewest directions, and every
named band beats the size-matched random control. Sweeping the k most
central directions (by |θ−45°|): central-k beats random-k at every
intermediate budget (max gap +0.093 P@1 at k = 75), and **k = 100 central
directions (33% of the space) retain 84% of full-space P@1** (0.696 vs
0.830). Language identification, by contrast, is near chance (0.51–0.57) in
*all* bands — post-alignment, language identity is not linearly decodable
from any band, an honest boundary of the method on MUSE-style spaces.

### 4.4 Relation offsets transfer by semantic type (Exp. 1, extending the
relation study)

Cross-language offset invariance orders exactly as intuition predicts
(cosine between r_pt and r_en; unrelated-offset baseline 0.00 ± 0.08):
capital–country 0.65 > gender 0.59 > plural 0.51 > antonyms 0.38. Feeding
the PT offsets into EN analogy queries (v_en(country) + r_pt ≈ v_en(capital),
NN over 20k):

| relation | top-1 | top-5 |
|---|---|---|
| masculine–feminine | 0.990 | 1.000 |
| capital–country | 0.711 | 0.942 |
| singular–plural | 0.282 | 0.480 |

World knowledge and lexicalized gender transfer almost losslessly;
morphology transfers poorly — consistent with plural offsets being the
least invariant. All relation offsets live slightly *closer to the shared
band* (mean θ 43–44°) than ordinary words (41.6°/42.5°): relations are more
language-neutral than the words they connect.

### 4.5 The shared band tracks language proximity (Exp. 4)

Same pipeline, four language pairs:

| pair | CKA | std θ | frac \|θ−45°\|≤5° | translation P@1 |
|---|---|---|---|---|
| PT×ES | **0.872** | 13.48° | 0.240 | **0.970** |
| PT×EN | 0.851 | 13.37° | 0.240 | 0.892 |
| EN×ES | 0.862 | 13.19° | 0.243 | 0.880 |
| EN×DE | 0.857 | **14.09°** | **0.220** | **0.800** |

EN×DE is unambiguously the narrowest frame (widest θ spread, smallest
shared band, worst retrieval); the sister pair PT×ES dominates CKA and
retrieval. The band fraction separates Germanic×Romance from the rest but
does not finely rank the three Romance-involving pairs — the spectrum is a
coarse typology signal at this vocabulary scale.

### 4.6 Non-aligned control: the spectrum detects absence of structure
(Exp. 6)

CommonCrawl PT × EN without alignment: CKA drops to 0.320, the θ spread
balloons (std 21.97° vs 13.37°), the shared band halves (0.130), translation
P@1 collapses to **0.007** (chance 0.0025), and direction-level translation
coherence is **0.000 everywhere**. The method does not hallucinate shared
directions where none exist.

### 4.7 mBERT layer-wise: the inverted-U, recovered per-direction (Exp. 5)

Extracting context-free word vectors from every layer of
bert-base-multilingual-cased and building one GSVD frame per layer: CKA,
shared-band fraction and retrieval all trace an inverted U with **layer 7 at
the peak** (frac 0.115, P@1 0.532; layer 8 second), embedding layer and
layer 12 worst — matching the literature's "middle layers are most
interlingual", but now with a per-direction spectrum rather than a scalar
probe. All mBERT layers remain far below supervised MUSE alignment
(shared-band 0.24, P@1 0.89): alignment does real geometric work that no
single static layer provides.

## 5. Negative results

We consider these load-bearing for the paper's credibility:

1. **Shared-band energy does not predict analogy success** (Exp. 1):
   correlation between an offset's energy in the 40–50° band and its
   transfer accuracy is *negative* (−0.45 for capitals; −0.22 pooled).
   Interpretation: what matters is the *type-level invariance* of the
   relation (§4.4), not where a particular offset's energy sits; energy
   concentration near 45° may simply flag noisy, low-norm offsets.
2. **Entities do not concentrate extreme energy at the word level**
   (Exp. 8): the small PT-side effect (0.445 vs 0.438 extreme-energy
   fraction, p = 0.0025) vanishes under frequency matching. The
   entity/common contrast is real at the *direction* level (extreme
   directions are name clusters, §4.2) but is not a per-word energy
   statistic — a useful caution against over-reading energy profiles.
3. **The Portuguese diminutive is not a linear direction, even in informal
   text** (Exp. 9): with CommonCrawl vectors Procrustes-mapped into the MUSE
   space (held-out cosine 0.72), diminutive offsets gain only marginal
   internal coherence (0.163 vs 0.142 on Wikipedia; random baseline −0.003)
   and their mean θ (42.8°) does not lean PT-ward. Productive morphology may
   require character-aware or context-aware representations to linearize.

## 6. Discussion

The through-line from MNIST to embeddings is that the GSVD frame is an
*instrument*, and θ is its readout. On low-overlap pairs (MNIST) the readout
is per-sample and discriminative. On high-overlap pairs (aligned languages)
the per-sample readout saturates — and the per-direction readout takes over,
ordering the space by specificity with exact algebra behind it (CᵀC + SᵀS =
I guarantees the cosine/sine pairing; no optimization, no probe training).
CKA tells you the two spaces agree at 0.85; the spectrum tells you *which*
24% of directions carry the agreement, what words live there, and what
functions (translation, relation transfer) that region supports.

## 7. Future work

- **Threshold-free classification.** Both image and embedding experiments
  show the 45° reference is a geometric landmark, not an optimal decision
  boundary; a likelihood-ratio or calibrated-posterior readout over the
  full θ(z) distribution should replace it.
- **Chinese and typologically distant pairs.** The en-zh vectors were
  unavailable at run time (server 403); segmentation-free scripts would
  stress the vocabulary-pairing step and test whether the shared band
  survives across scripts.
- **Contextual tokens in context.** Exp. 5 used words in isolation;
  repeating with in-context token embeddings (and per-layer *sentence*
  frames) would test whether context widens the shared band and whether the
  θ of a token moves with code-switching.
- **Beyond pairs: n-ary frames.** The GSVD is binary; HO-GSVD or sequential
  frames (PT×EN band, then band×ES) could produce a multilingual spectrum.
- **Learned features for images.** The CIFAR-10 saturation at the pixel
  level invites re-running the image experiments on pretrained-CNN features,
  paralleling the raw-vs-aligned contrast of §4.6.
- **A better morphology probe.** The diminutive negative (Exp. 9) motivates
  subword-aware embeddings or minimal-pair contexts as the right
  representation for testing whether productive morphology has a direction.
- **Theory.** The central-band retrieval result (84% of P@1 in 33% of the
  directions, always above random) suggests a formal statement relating
  θ-centrality to cross-dataset reconstruction error; the CS-decomposition
  structure should make this provable.

## Reproducibility

All experiments run from `paper_explorations.ipynb` on cached public data
(~200 MB total); the GSVD implementation is validated against Julia/LAPACK
to 1e-13 (factor-level equivalence suite). Seeds are fixed; every table in
this draft is emitted to `examples/paper_results/*.json` by the notebook.
