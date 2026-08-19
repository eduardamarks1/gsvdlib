"""gsvdlib: GSVD-based geometry-grounded dataset comparison.

Python port of the Julia pipeline from *GSVD for Geometry-Grounded Dataset
Comparison: An Alignment Angle Is All You Need*. The cosine factor is named
``C`` (formerly D1) and the sine factor ``S`` (formerly D2), as in the paper.
"""

from .angles import (cosine_similarity, get_angles_C, get_angles_S,
                     get_most_similar_row_H, get_nonzero_per_column,
                     highlight_closest)
from .blocks import infer_block_sizes, to_intersection, zero_pure_blocks
from .classify import (classify, classify_set, linear_cka,
                       metrics_from_angles, theta_angles)
from .core import (GSVDResult, gsvd, make_A_B, make_C_S, permutation,
                   set_to_0_closest_to_45, sort_and_rebuild, wire_size)
from .datasets import (ArrayDataset, FashionMNISTDataset, MNISTDataset,
                       VectorDataset, balanced_count, center, sample_pair)
from .lapack_gsvd import GeneralizedFactors, gsvd_factors
from .pipeline import (PreparedPair, evaluate_pair, prepare_data,
                       run_pair_experiment)

__version__ = "0.1.0"

__all__ = [
    "gsvd", "GSVDResult", "gsvd_factors", "GeneralizedFactors",
    "make_C_S", "make_A_B", "wire_size", "permutation",
    "sort_and_rebuild", "set_to_0_closest_to_45",
    "infer_block_sizes", "zero_pure_blocks", "to_intersection",
    "get_nonzero_per_column", "get_angles_C", "get_angles_S",
    "highlight_closest", "cosine_similarity", "get_most_similar_row_H",
    "theta_angles", "classify", "classify_set", "metrics_from_angles",
    "linear_cka",
    "VectorDataset", "ArrayDataset", "MNISTDataset", "FashionMNISTDataset",
    "sample_pair", "balanced_count", "center",
    "prepare_data", "evaluate_pair", "run_pair_experiment", "PreparedPair",
]
