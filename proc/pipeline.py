import pandas as pd
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
from functools import partial, reduce

from proc.pre import (
    load,
    latitude_filter,
    longitude_filter,
    select_columns,
    drop_rows_with_nan_values,
    add_sgr_pm_correction,
    sample,
    create_colors,
)

from sgrmemb.reprojection import compute_Sgr_XY
from sgrmemb.sgr_coords import compute_Sgr_coords

"""
In this file all the preprocessing functions are patched together.
"""

COLS_TO_KEEP = [
    "ra",
    "dec",
    "l",
    "b",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "parallax",
    "parallax_error",
    "mag_J",
    "er_J",
    "mag_H",
    "er_H",
    "mag_Ks",
    "er_Ks",
    "phot_g_mean_mag",
    "bp_rp",
    #"prob_xi_memb",  TODO: uncomment
    #"prob_xi_field",  TODO: uncomment
]

COLS_WITHOUT_NANS = [
    "ra",
    "dec",
    "l",
    "b",
    "mag_J",
    "mag_H",
    "mag_Ks",
    #"prob_xi_memb",  TODO: uncomment
    #"prob_xi_field",  TODO: uncomment
]

SPACE_PARAMS = {
    "pos_αδ": ["ra", "dec"],
    "pos_lb": ["l", "b"],
    "pos_xy": ["SgrX", "SgrY"],
    "pm": ["pmra", "pmdec"],
    "pm_corr": ["pmra_corr", "pmdec_corr"],
    "cmd": ["mag_J-mag_Ks", "mag_Ks"],
    "cmd_gaia": ["bp_rp", "phot_g_mean_mag"],
}


def get_subspace_data(
    df: pd.DataFrame,
    space_params: dict[str, str],
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Return an object which stores the data of each respective sub-space.
    """
    output = dict()
    for key, value in space_params.items():
        output[key] = df[value].to_numpy()

    return output


def load_and_preproc(
    filepath: Path,
    bmin: Optional[float] = None,
    bmax: Optional[float] = None,
    lmin: Optional[float] = None,
    lmax: Optional[float] = None,
    sample_size: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[str, npt.NDArray[np.float64]]]:
    """
    Read the data from the input file and return an object which stores the data of each
    respective sub-space.
    """

    PreprocFunction = Callable[[pd.DataFrame], pd.DataFrame]

    def _compose(*functions: PreprocFunction) -> PreprocFunction:
        """
        A function composer, that is, it returns a function which is the composition
        of the input functions.
        Example:
        ``(f o g o h)(x) = f(g(h(x))``
        where f, g and h are functions, and x is the input.
        """
        return reduce(lambda f, g: lambda x: g(f(x)), functions)

    preproc_pipeline = [
        partial(latitude_filter, bmin=bmin, bmax=bmax),
        partial(longitude_filter, lmin=lmin, lmax=lmax),
        partial(select_columns, cols_to_keep=COLS_TO_KEEP),
        partial(drop_rows_with_nan_values, cols_to_check=COLS_WITHOUT_NANS),
        partial(sample, sample_size=sample_size),
        compute_Sgr_XY,
        compute_Sgr_coords,
        add_sgr_pm_correction,
        partial(create_colors, colors=["mag_J-mag_Ks", "mag_H-mag_Ks"]),
    ]

    df = load(filepath)
    preprocess = _compose(*preproc_pipeline)
    df = preprocess(df)

    # Generate a dictionary with numpy arrays ready to be processed
    outdict = get_subspace_data(df, SPACE_PARAMS)

    return df, outdict
