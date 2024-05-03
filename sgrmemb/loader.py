from functools import partial, reduce
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import astropy.coordinates as coord
import numpy as np
import numpy.typing as npt

import pandas as pd

from sgrmemb import preproc


COLS_TO_KEEP = [
    "ra",
    "dec",
    "l",
    "b",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "mag_J",
    "er_J",
    "mag_H",
    "er_H",
    "mag_Ks",
    "er_Ks",
    "phot_g_mean_mag",
    "bp_rp",
]

SPACE_PARAMS = {
    "pos_αδ": ["ra", "dec"],
    "pos_lb": ["l", "b"],
    "pos_xy": ["SgrX", "SgrY"],
    "pm": ["pmra", "pmdec"],
    "pm_corr": ["pmra_corr", "pmdec_corr"],
    "bands": ["mag_J", "mag_H", "mag_Ks"],
    "cmd": ["mag_J-mag_Ks", "mag_Ks"],
    "cmd_hk": ["mag_H-mag_Ks", "mag_Ks"],
    "cmd_jhk": ["mag_J-mag_Ks", "mag_H"],
    "ccd": ["mag_J-mag_Ks", "mag_H-mag_Ks"],
    "cmd_gaia": ["bp_rp", "phot_g_mean_mag"],
}

def load(filepath: Path) -> pd.DataFrame:
    """
    Read the data from the input file.
    It is expected a csv file with the columns names
    or a fits table
    """

    suffix = filepath.suffix

    if suffix == ".fits":
        from astropy.table import Table

        table = Table.read(filepath, format="fits")
        df = table.to_pandas()

    elif suffix == ".csv":
        df = pd.read_csv(filepath)

    else:
        raise ValueError(f"File extension {suffix} not recognized")

    return df


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


def get_arrays_std(filepath: Path) -> tuple[pd.DataFrame, dict[str, npt.NDArray[np.float64]]]:
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

    std_pipeline = [
        partial(preproc.filter_df_columns, cols_to_keep=COLS_TO_KEEP),
        partial(preproc.drop_rows_with_nan_values, cols_to_check=COLS_TO_KEEP),
        preproc.compute_Sgr_XY,
        preproc.add_sgr_pm_correction,
        partial(preproc.create_colors, colors=["mag_J-mag_Ks", "mag_H-mag_Ks"]),
        partial(preproc.remove_fainter_than, mag_cut=17.0, band="mag_Ks"),
    ]

    logging.info(f"Perfoming standard preprocessing {filepath.name}")
    df = load(filepath)
    preprocess = _compose(*std_pipeline)
    df = preprocess(df)
    outdict = get_subspace_data(df, SPACE_PARAMS)
    return df, outdict


def get_arrays_vasiliev(filepath: Path) -> dict[str, npt.NDArray[np.float64]]:
    """
    Read the data from the input file and return an object which stores the data of each
    respective sub-space.
    Notice that the data is resampled to have the number of stars in each bin proportional
    to the mean probability of the bin, following instructions from the paper.
    """
    logging.info(f"Getting arrays from vasiliev data {filepath.name}")

    df_catalogue = pd.read_csv(filepath, sep=" ")
    df_low = df_catalogue.query("memberprob < 0.5").sample(frac=0.5)
    df_high = df_catalogue.query("memberprob >= 0.5")

    # Resample df_high
    bins_range = np.linspace(0.5, 1.0, 51)
    column_name = "memberprob"  # Column to bin

    # Create bins
    bins = pd.cut(df_high[column_name], bins=bins_range, labels=False)
    df_high_resampled = (
        df_high.groupby(bins)
        .apply(lambda x: x.sample(frac=x.memberprob.max()))
        .reset_index(drop=True)
    )

    # concatenate the two dataframes
    df = pd.concat([df_low, df_high_resampled])

    COLS = [
        "ra",
        "dec",
        "pmra",
        "pmdec",
        "g_mag",
        "bp_rp",
        "memberprob",
    ]
    df = df[COLS].dropna()
    df = preproc.compute_Sgr_XY(df)

    # Generate galactic coordinates
    coords_radec = coord.SkyCoord(ra=df["ra"], dec=df["dec"], unit="deg")
    df["l"] = coords_radec.galactic.l.value
    df["b"] = coords_radec.galactic.b.value

    df = preproc.add_sgr_pm_correction(df)
    return get_subspace_data(
        df,
        {
            "pos_xy": ["SgrX", "SgrY"],
            "pos_lb": ["l", "b"],
            "cmd": ["bp_rp", "g_mag"],
            "ast": ["pmra", "pmdec"],
        },
    )
