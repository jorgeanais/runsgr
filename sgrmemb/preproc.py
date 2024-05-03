from typing import Callable, Optional

import pandas as pd
from sgrmemb.reprojection import compute_Sgr_XY


def remove_fainter_than(df: pd.DataFrame, mag_cut: float, band: str) -> pd.DataFrame:
    """
    Remove stars fainter than mag in the band band.
    """
    return df[df[band] <= mag_cut]


def add_sgr_pm_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Correct proper motions based on Vasiliev et al. 2020"""

    df = df.copy()

    SGR_RA = 283.764
    SGR_DEC = -30.480
    SGR_MU_RA = -2.692
    SGR_MU_DEC = -1.359

    ra_1 = df["ra"].to_numpy()
    dec_1 = df["dec"].to_numpy()

    delta_ra = ra_1 - SGR_RA
    delta_dec = dec_1 - SGR_DEC

    new_mu_alpha = (
        -2.69
        + 0.009 * delta_ra
        - 0.002 * delta_dec
        - 0.00002 * delta_ra * delta_ra * delta_ra
    )
    new_mu_delta = (
        -1.35
        - 0.024 * delta_ra
        - 0.019 * delta_dec
        - 0.00002 * delta_ra * delta_ra * delta_ra
    )

    df["pmra_corr"] = new_mu_alpha
    df["pmdec_corr"] = new_mu_delta

    return df


def create_colors(
    df: pd.DataFrame, colors: list[str] = ["mag_J-mag_Ks"]
) -> pd.DataFrame:
    """
    Create colors from the magnitudes. The colors are added as new columns to the dataframe.
    """

    df = df.copy()

    for color in colors:
        band1, band2 = color.split("-")
        df[color] = df[f"{band1}"] - df[f"{band2}"]

    return df


def filter_df_columns(df: pd.DataFrame, cols_to_keep: list[str]) -> pd.DataFrame:
    """
    Keep only specified columns.
    """
    df = df.copy()
    return df[cols_to_keep]


def drop_rows_with_nan_values(df: pd.DataFrame, cols_to_check: list[str]) -> pd.DataFrame:
    """
    Remove rows with NaN values in the specified columns.
    """
    df = df.copy()
    return df.dropna(subset=cols_to_check)
