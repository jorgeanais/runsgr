import logging
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from cpsplines.fittings.fit_cpsplines import CPsplines
from cpsplines.graphics.plot_curves import CurvesDisplay

from sgrmemb.plots import plot_gmm_results
from sgrmemb.settings import FileConfig, ParamsConfig


Q_FACTOR = (
    1.0 / 0.6
)  # From the isophote analysis of the vasiliev data (see notebook deprojection/ellipse.ipynb)


class DistributionFunction:
    def __call__(self, X: npt.NDArray) -> npt.NDArray:
        ...


def compute_elliptical_radius(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    q_factor: float = Q_FACTOR,
) -> npt.ArrayLike:
    """
    Compute the elliptical radius of a point (x, y) with a given q_factor.
    where x is the semi-major axis and y is the semi-minor axis.
    """
    return np.sqrt(np.power(x, 2) + np.power(q_factor * y, 2))


def get_radial_density_estimation(
    r: npt.ArrayLike,
    weights: npt.ArrayLike,
    n_bins: int = 25,  #TODO: What is the impact of this parameter on the final distribution?
    scale: Literal["linear", "log"] = "linear",
    min_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute the radial density estimation per unit of area of a set of points.
    """

    # Check input parameters
    if min_radius is None:
        min_radius = np.min(r)

    if max_radius is None:
        max_radius = np.max(r)

    if scale == "linear":
        bins = np.linspace(start=min_radius, stop=max_radius, num=n_bins)
    elif scale == "log":
        if min_radius <= 0:
            raise ValueError(
                f"min_radius must be greater than 0 for log scale. {min_radius=}"
            )
        bins = np.logspace(
            start=np.log10(min_radius), stop=np.log10(max_radius), num=n_bins
        )
    else:
        raise ValueError(f"Invalid scale type: {scale}. Options are 'linear' or 'log'")

    # Construct the histogram
    counts, bin_edges = np.histogram(
        r,
        bins=bins,
        weights=weights,
    )

    # Double the contribution of stars with r>=8 deg to compensate for the missing area # TODO: check this
    indx = np.where(bin_edges >= 8.0)[0]
    counts[indx - 1] = 2 * counts[indx - 1]

    area = 0.25 * np.pi * np.diff(bin_edges**2)
    counts_ = counts / area  # Counts per unit of area
    mid_points = (bin_edges[1:] + bin_edges[:-1]) / 2

    dataframe = pd.DataFrame(
        {
            "mid_points": mid_points if scale == "linear" else np.log10(mid_points),
            "counts": counts_ if scale == "linear" else np.log10(counts_),
        }
    )

    # Remove -inf values
    # dataframe = dataframe.replace(-np.inf, np.nan).dropna()  # TODO: check if this line is necessary

    return dataframe


def cubic_spline_fitting(
    df: pd.DataFrame,
    fit_family: Literal["poisson", "gaussian"] = "poisson",
    x_col: str = "mid_points",
    y_col: str = "counts",
    plot: bool = False,
    x_range: Optional[tuple[float, float]] = None,
    n_intervals: int = 2,
) -> CPsplines:
    """
    Fit a cubic spline model to the radial density estimation in a defined range.
    x_range: an optional tuple containing the most extreme values that the extended
             B-spline basis needs to capture along each dimension
    n_intervals: number of equal intervals which the fitting region along each dimension is split
    """

    if n_intervals < 1:
        raise ValueError(f"n_int must be greater than 1. {n_intervals=}")

    spline_model = CPsplines(
        deg=(3,),
        ord_d=(2,),
        n_int=(n_intervals,),
        x_range={x_col: x_range},
        family=fit_family,
    )

    spline_model.fit(data=df, y_col=y_col)

    if plot:
        p = CurvesDisplay.from_estimator(
            spline_model,
            X=df[x_col],
            y=df[y_col],
            knot_positions=True,
            col_pt="b",
            density=10,
            **{"c": "g"},
        )
        _ = p.ax_.set_xlabel("x", fontsize=16)
        _ = p.ax_.set_ylabel("Weighted counts", fontsize=16)
        _ = p.ax_.tick_params(axis="both", which="major", labelsize=16)
        if x_range is not None:
            _ = p.ax_.axvline(x=x_range[0], color="r", linestyle="--")
            _ = p.ax_.axvline(x=x_range[1], color="r", linestyle="--")
        plt.show()
        plt.clf()
        plt.close()

    return spline_model


def get_normalization_constant(
    target_func: DistributionFunction,
    region: tuple[float, float],
    samples=10_000,
) -> float:
    """
    Compute the normalization constant of the distribution function.
    If mask_value is provided, all values below this threshold are evaluated as mask_value.
    """

    x = np.linspace(start=region[0], stop=region[1], num=samples, endpoint=True)
    y_pred = target_func(x)
    norm_constant = np.trapz(y_pred, x)

    return norm_constant


def gen_ast(
    X_train: npt.NDArray,
    auto_identify: bool = True,
    plot: bool = True,
    n_components: int = 3,
) -> tuple[DistributionFunction, DistributionFunction]:

    """Generate the distribution function asociated to the astrometric data."""

    from sklearn.mixture import GaussianMixture
    from scipy.stats import multivariate_normal

    logging.info("Generating astrometric distribution function")

    MEMBER_INDEX = 1

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=0
    ).fit(X_train)

    # Identify narrowest component
    if auto_identify:
        covariances = gmm.covariances_
        eigenvalues = [np.linalg.eigvals(cov) for cov in covariances]
        narrowest_index = np.argmin([min(eigvals) for eigvals in eigenvalues])
        MEMBER_INDEX = narrowest_index

    if plot:
        plot_gmm_results(
            X_train,
            means=gmm.means_,
            covariances=gmm.covariances_,
            title="Proper Motion model",
            zoom_region=[[-15, 10], [-15, 10]],
            save_path=FileConfig.OUTPUT_PATH / "gmm_pm.png",
        )
        logging.info(f"{MEMBER_INDEX=}")

    sgr_indx = np.array([MEMBER_INDEX])
    field_indx = np.setdiff1d(np.arange(n_components), sgr_indx)

    field_means = gmm.means_[field_indx]
    field_covariances = gmm.covariances_[field_indx]
    field_weights = gmm.weights_[field_indx]

    memb_means = gmm.means_[sgr_indx]
    memb_covariances = gmm.covariances_[sgr_indx]
    memb_weights = gmm.weights_[sgr_indx]

    # Normalization
    field_weights = field_weights / np.sum(field_weights)
    memb_weights = memb_weights / np.sum(memb_weights)

    def pdf_field(X: npt.NDArray) -> npt.NDArray:
        """Return the field probability density function of the field model at X"""
        return np.sum(
            [
                w * multivariate_normal.pdf(X, mean=mean, cov=cov)
                for w, mean, cov in zip(field_weights, field_means, field_covariances)
            ],
            axis=0,
        )

    def pdf_memb(X: npt.NDArray) -> npt.NDArray:
        """Return the Sgr  probability density function of the field model at X"""
        return np.sum(
            [
                w * multivariate_normal.pdf(X, mean=mean, cov=cov)
                for w, mean, cov in zip(memb_weights, memb_means, memb_covariances)
            ],
            axis=0,
        )

    return pdf_memb, pdf_field





def gen_ast_memb(
    X_train: npt.NDArray, weights: npt.NDArray, plot: bool = False
) -> DistributionFunction:
    """Generate the distribution function asociated to the astrometric data for members only."""

    from scipy.stats import multivariate_normal
    from sklearn.mixture import GaussianMixture

    COV_FACTOR: float = 1.0
    N_COMPONENTS: int = 1

    indx_memb = np.where(weights > 0.5)  # TODO: check this (how justify this value?)

    gmm = GaussianMixture(
        n_components=N_COMPONENTS, covariance_type="full", random_state=0
    ).fit(X_train[indx_memb])

    if plot:
        plot_gmm_results(
            X_train[indx_memb],
            means=gmm.means_,
            covariances=gmm.covariances_,
            title="Proper Motion model",
            zoom_region=[[-3.5, -2], [-2, 0]],
        )

    # sgr_indx = np.array([0])
    memb_means = gmm.means_  # [sgr_indx]
    memb_covariances = gmm.covariances_  # [sgr_indx]
    memb_weights = gmm.weights_  # [sgr_indx]

    def pdf_memb(X: npt.NDArray) -> npt.NDArray:
        """Return the Sgr  probability density function of the field model at X"""
        return np.sum(
            [
                1.0 * multivariate_normal.pdf(X, mean=mean, cov=COV_FACTOR * cov)
                for w, mean, cov in zip(memb_weights, memb_means, memb_covariances)
            ],
            axis=0,
        )

    return pdf_memb


def gen_ast_memb2(
    X_train: npt.NDArray,
    weights: npt.NDArray,
    plot: bool = False,
    factor_cov: float = 1.0,
) -> DistributionFunction:
    """Generate the distribution function asociated to the astrometric data for members only."""

    from scipy.stats import multivariate_normal

    mean = np.average(X_train, weights=weights, axis=0)
    cov = factor_cov * np.cov(X_train.T, aweights=weights)

    logging.info(f"{mean=}")
    logging.info(f"{cov=}")

    def pdf_memb(X: npt.NDArray) -> npt.NDArray:
        """Return the Sgr  probability density function of the field model at X"""
        return multivariate_normal.pdf(X, mean=mean, cov=cov)

    return pdf_memb




def gen_pos_memb(
    X_train: npt.NDArray,
    weights: Optional[npt.NDArray] = None,
    n_bins: int = 25,  # what is the impact of this parameter on the final distribution?
    min_fit_r: Optional[float] = None,
    max_fit_r: Optional[float] = None,
    r_mask_threshold: Optional[float] = 0.15,
    n_intervals: int = 10,
    plot: bool = False,
) -> DistributionFunction:
    """
    Generate the distribution function asociated with positional data for Sgr members.
    Notice that internal calculations are done in log space.

    X_train: array of shape (n_samples, 2) with the positional data
    weights: array of shape (n_samples,) with the weights of each sample
    n_bins: number of bins to use in the radial density estimation
    min_fit_r: minimum radius to use in the fitting region
    max_fit_r: maximum radius to use in the fitting region
    r_mask_threshold: threshold to flatten out the data in the center of the distribution
    n_intervals: number of equal intervals which the fitting region along each dimension is split
    """

    logging.info("Generating the distribution function for memb positional data")

    r_ell = compute_elliptical_radius(X_train[:, 0], X_train[:, 1], Q_FACTOR)
    r_ell_min = np.min(r_ell)
    r_ell_max = np.max(r_ell)

    if weights is None:
        weights = np.ones_like(r_ell)

    # Check boarders of the fitting region
    if min_fit_r is None:
        min_fit_r = max(r_ell_min, 0.0001)
    elif min_fit_r <= 0:
        raise ValueError("min_fit_r must be greater than 0")

    if max_fit_r is None:
        max_fit_r = r_ell_max
    elif max_fit_r < min_fit_r:
        raise ValueError("max_fit_r must be greater than min_fit_r")

    df = get_radial_density_estimation(
        r=r_ell,
        weights=weights,
        min_radius=min_fit_r,
        max_radius=max_fit_r,  # TODO: Check thisn_bins: int = 25
        n_bins=n_bins,
        scale="log",
    )

    spline_model = cubic_spline_fitting(
        df=df,
        plot=plot,
        x_range=(np.log10(r_ell_min), np.log10(r_ell_min)),
        n_intervals=n_intervals,
    )

    def candidate_function(r: npt.NDArray) -> npt.NDArray:
        """Spline model before normalization. Central region can be masked to a constant value."""

        # Mask the central region of the function
        if r_mask_threshold is not None:
            r = np.where(r < r_mask_threshold, r_mask_threshold, r)

        return np.power(10, spline_model.predict(pd.Series(np.log10(r))))

    norm_constant = get_normalization_constant(
        target_func=candidate_function,
        region=(r_ell_min, r_ell_max),
        samples=30_000,
    )

    def pdf_elliptical_model(X: npt.NDArray) -> npt.NDArray:
        """Return the probability density function of the elliptical model at X"""
        r = compute_elliptical_radius(X[:, 0], X[:, 1], q_factor=Q_FACTOR)
        return candidate_function(r) / norm_constant

    return pdf_elliptical_model


def gen_pos_field(
    X_train: npt.NDArray,
    weights: Optional[npt.NDArray] = None,
    n_bins: int = 10,
    n_intervals: int = 2,
) -> DistributionFunction:
    """
    Generate the distribution function asociated with positional data for field objects.

    Parameters
    ----------
    X_train: array of shape (n_samples, 2) with the positional data (l, b).
    weights: array of shape (n_samples,) with the weights of each sample
    n_bins: number of bins to use in the density estimation
    n_intervals: number of intervals to use in the cubic spline fitting

    Returns
    -------
    pdf_field: function that returns the fitted probability density function of the field model
    """

    logging.info("Generating the distribution function for field positional data")

    b_data = X_train[:, 1]
    b_range = (b_data.min(), b_data.max())

    if weights is None:
        weights = np.ones_like(b_data)

    counts, bin_edges = np.histogram(
        b_data, bins=n_bins, range=b_range, weights=weights
    )
    mid_points = (bin_edges[1:] + bin_edges[:-1]) / 2
    df = pd.DataFrame({"mid_points": mid_points, "counts": counts})

    spline_model = cubic_spline_fitting(
        df=df,
        x_range=b_range,
        n_intervals=n_intervals,
    )

    norm_constant = get_normalization_constant(
        target_func=lambda x: spline_model.predict(pd.Series(x)),
        region=b_range,
        samples=100_000,
    )

    def df_galactic_latitude_gradient(X: npt.NDArray) -> npt.NDArray:
        """Return the probability density function of the positional field model for galactic latitud"""
        lat_b = X[:, 1]
        return spline_model.predict(pd.Series(lat_b)) / norm_constant

    return df_galactic_latitude_gradient


def gen_phot(X_train: npt.NDArray, weights: npt.NDArray) -> DistributionFunction:
    """Generate the distribution function asociated with photometric data."""

    from scipy.ndimage import gaussian_filter

    logging.info("Generating the distribution function for photometric data")

    BIN_SIZE = 0.05
    SIGMA = 0.7  # 0.5 -> 1.0

    # Borders of the CMD
    xmin = X_train[:, 0].min()
    xmax = X_train[:, 0].max()
    ymin = X_train[:, 1].min()
    ymax = X_train[:, 1].max()

    # Number of bins
    binx = int(np.ceil((xmax - xmin) / BIN_SIZE))
    biny = int(np.ceil((ymax - ymin) / BIN_SIZE))

    # Compute histogram
    hist, xedges, yedges = np.histogram2d(
        X_train[:, 0], X_train[:, 1], bins=(binx, biny), weights=weights
    )

    # Smooth the histogram using a Gaussian kernel
    smoothed_hist = gaussian_filter(hist, sigma=SIGMA)

    # Normalize the histogram
    smoothed_hist = smoothed_hist / np.sum(smoothed_hist)

    def cmd_pdf(X: npt.NDArray) -> npt.NDArray:
        """Compute the index of the bin where the point is located"""
        indx_x = np.digitize(X[:, 0], xedges, right=False)
        indx_y = np.digitize(X[:, 1], yedges, right=False)

        # Check if the point is out of the histogram TODO: check if this is correct
        shape_hist = smoothed_hist.shape
        indx_x_c = np.where(indx_x >= shape_hist[0], shape_hist[0] - 1, indx_x)
        indx_y_c = np.where(indx_y >= shape_hist[1], shape_hist[1] - 1, indx_y)

        # Compute the probability of each point
        pdf = smoothed_hist[indx_x_c, indx_y_c]

        return pdf

    return cmd_pdf
