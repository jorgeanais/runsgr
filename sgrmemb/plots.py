import itertools
from pathlib import Path
from typing import Iterable, Optional

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from scipy import linalg

COLOR_ITER = itertools.cycle(
    [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
    ]
)


def plot_gmm_results(
    X: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    covariances: npt.NDArray[np.float64],
    title: str = "GMM fit",
    invert_yaxis: bool = False,
    zoom_region: list[list[float]] = [[-15.0, 10.0], [-15.0, 10.0]],
    alpha_ell: float = 0.2,
    save_path: Optional[Path] = None,
):
    splot = plt.subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], 0.8, color="black", alpha=0.1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, COLOR_ITER)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(alpha_ell)
        splot.add_artist(ell)

        # Add label with id
        plt.text(x=mean[0], y=mean[1], s=str(i), fontdict={"fontsize": 8})

    # plt.xlim(-6.0, 4.0 * np.pi - 6.0)
    # plt.ylim(19, 11)
    plt.title(title)
    if invert_yaxis:
        plt.gca().invert_yaxis()
    if zoom_region is not None:
        plt.xlim(zoom_region[0])
        plt.ylim(zoom_region[1])

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def histogram_2d(
    X: npt.NDArray[np.float64],
    ax: plt.Axes,
    bin_size_x: float = 0.1,
    bin_size_y: float = 0.1,
    gamma_value: float = 0.5,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    invert_xaxis: bool = False,
    invert_yaxis: bool = False,
    equal_aspect: bool = False,
) -> ScalarMappable:
    """
    Creates a 2D histogram

    Parameters
    ----------
    X : npt.NDArray[np.float64]
        The data to plot
    ax : plt.Axes
        The axis to plot the data
    bin_size_x : float, optional
        The size of the bins in the x-axis, by default 0.1
    bin_size_y : float, optional
        The size of the bins in the y-axis, by default 0.1
    gamma_value : float, optional
        The gamma value for the colorbar normalization, by default 0.5
    xlim : Optional[tuple[float, float]], optional
        The limits of the x-axis, by default None
    ylim : Optional[tuple[float, float]], optional
        The limits of the y-axis, by default None
    title : Optional[str], optional
        The title of the plot, by default None
    invert_xaxis : bool, optional
        Invert the x-axis, by default False
    invert_yaxis : bool, optional
        Invert the y-axis, by default False
    equal_aspect : bool, optional
        Set the aspect ratio to be equal, by default False
    """

    if title:
        ax.set_title(title)

    if xlim is None:
        xlim = (X[:, 0].min(), X[:, 0].max())

    if ylim is None:
        ylim = (X[:, 1].min(), X[:, 1].max())

    # Compute number of bins in each axis
    binx = int(np.ceil((xlim[1] - xlim[0]) / bin_size_x))
    biny = int(np.ceil((ylim[1] - ylim[0]) / bin_size_y))

    # https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot
    counts, xedges, yedges, im = ax.hist2d(
        X[:, 0].flatten(),
        X[:, 1].flatten(),
        bins=(binx, biny),
        range=[xlim, ylim],
        norm=mcolors.PowerNorm(gamma_value),
    )

    if invert_xaxis:
        ax.invert_xaxis()

    if invert_yaxis:
        ax.invert_yaxis()

    if equal_aspect:
        ax.set_aspect("equal")

    return im


def support_summary(
    edges: npt.NDArray[np.float64],
    supports: npt.NDArray[np.float64],
    percentages: npt.NDArray[np.float64],
    indx: int,
) -> str:
    """
    Creates a string with the support summary for a given slice

    Parameters
    ----------
    edges : npt.NDArray[np.float64]
        The edges of the probability slices
    supports : npt.NDArray[np.float64]
        The support of each slice (counts)
    percentages : npt.NDArray[np.float64]
        The percentage of each slice (counts/N*100)
    indx : int
        The index of the slice
    """
    return f"p={edges[indx-1]:.2f}-{edges[indx]:.2f} s={supports[indx-1]}({percentages[indx-1]:.1f}%)"


def plot_sgr_stream_nom_position(ax: plt.Axes) -> None:
    """Plot the nominal position of the Sgr stream on the given axis."""

    from sgrmemb.sgr_coords import sgr_stream_gal

    ax.plot(sgr_stream_gal.l.deg, sgr_stream_gal.b.deg, c="red", linestyle="dashed")


def plot_pos_lb(
    X: npt.NDArray[np.float64],
    ax: plt.Axes,
    fig: plt.Figure,
    title: Optional[str] = None,
) -> None:
    """
    Plot the positions of the stars.

    Parameters
    ----------
    X : npt.NDArray[np.float64]
        The positions of the stars (preferentially in galactic coordinates)
    ax : plt.Axes
        The axes to plot the data
    fig : plt.Figure
        The figure to plot the data
    title : str, optional
        The title of the plot (preferentially the support_summary), by default None
    """
    map_pos = histogram_2d(
        X=X,
        ax=ax,
        bin_size_x=0.1,
        bin_size_y=0.1,
        gamma_value=0.5,
        xlim=None,
        ylim=None,
        title=title,
        invert_xaxis=True,
        invert_yaxis=False,
        equal_aspect=False,
    )
    fig.colorbar(map_pos, ax=ax)


def plot_proper_motions(
    X: npt.NDArray[np.float64], ax: plt.Axes, fig: plt.Figure
) -> None:
    """
    Plot the proper motions of the stars.

    Parameters
    ----------
    X : npt.NDArray[np.float64]
        The proper motions of the stars (preferentially in galactic coordinates)
    ax : plt.Axes
        The axes to plot the data
    fig : plt.Figure
        The figure to plot the data
    """
    map_ast = histogram_2d(
        X=X,
        ax=ax,
        bin_size_x=0.3,
        bin_size_y=0.3,
        gamma_value=0.5,
        xlim=(-20, 10),
        ylim=(-20, 10),
        title=None,
        invert_xaxis=False,
        invert_yaxis=False,
        equal_aspect=False,
    )
    fig.colorbar(map_ast, ax=ax)


def plot_color_magnitude_diagram(
    X: npt.NDArray[np.float64], ax: plt.Axes, fig: plt.Figure
):
    """
    Plot a color-magnitude diagram.

    Parameters
    ----------
    X : npt.NDArray[np.float64]
        The photometry of the stars (Color expected at X[:, 0] and magnitude at X[:, 1])
    ax : plt.Axes
        The axes to plot the data
    fig : plt.Figure
        The figure to plot the data
    """
    map_cmd = histogram_2d(
        X=X,
        ax=ax,
        bin_size_x=0.05,
        bin_size_y=0.05,
        gamma_value=0.5,
        xlim=(-0.1, 2.1),
        ylim=None,
        title=None,
        invert_xaxis=False,
        invert_yaxis=True,
        equal_aspect=False,
    )
    fig.colorbar(map_cmd, ax=ax)


def plot_color_color_diagram(X: npt.NDArray[np.float64], ax: plt.Axes, fig: plt.Figure):
    """
    Plot a color-color diagram.

    Parameters
    ----------
    X : npt.NDArray[np.float64]
        The colors of the stars (x-axis and y-axis colors are expected at X[:, 0] and X[:, 1] respectively)
    ax : plt.Axes
        The axes to plot the data
    fig : plt.Figure
        The figure to plot the data
    """
    map_ccd = histogram_2d(
        X=X,
        ax=ax,
        bin_size_x=0.03,
        bin_size_y=0.03,
        gamma_value=0.2,
        xlim=(-0.1, 1.5),
        ylim=(-0.1, 0.6),
        title=None,
        invert_xaxis=False,
        invert_yaxis=False,
        equal_aspect=False,
    )
    fig.colorbar(map_ccd, ax=ax)


def make_prob_slices_plots(
    X_pos: npt.NDArray[np.float64],
    X_ast: npt.NDArray[np.float64],
    X_cmd: npt.NDArray[np.float64],
    prob: npt.NDArray[np.int64],
    n_slices: int = 6,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the phase-space according for slices of membership probability"""

    # Make probability slices
    xedges = np.linspace(0.0, 1.0, n_slices + 1)
    corresponding_slice = np.digitize(prob, xedges)
    unique_slices, slice_support = np.unique(corresponding_slice, return_counts=True)
    slice_percentage = slice_support / len(prob) * 100

    # Plot the data
    n_rows = 3  # One for each parameter (pos, ast, cmd)
    n_cols = n_slices  # number of columns
    n_subplots = n_rows * n_cols  # total number of subplots

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Generate phase-space plots for each slice
    for column_indx in range(n_cols):

        # Data selection according to the current probability slice
        slice_indx = column_indx + 1
        indx_sel = np.where(corresponding_slice == slice_indx)

        # Plot the positional data
        pos_ax = axes[0, column_indx]
        support_summary_ = support_summary(
            edges=xedges,
            supports=slice_support,
            percentages=slice_percentage,
            indx=slice_indx,
        )
        plot_pos_lb(
            X=X_pos[indx_sel],
            ax=pos_ax,
            fig=fig,
            title=support_summary_,
        )
        plot_sgr_stream_nom_position(ax=pos_ax)

        # Plot the proper motions
        ast_ax = axes[1, column_indx]
        plot_proper_motions(X=X_ast[indx_sel], ax=ast_ax, fig=fig)

        # Plot the CMD
        cmd_ax = axes[2, column_indx]
        plot_color_magnitude_diagram(X=X_cmd[indx_sel], ax=cmd_ax, fig=fig)

        if save_path is not None:
            fig.savefig(save_path, dpi=300)


# def make_prob_slices_plots_ccd(
#     X_pos: npt.NDArray[np.float64],
#     X_ast: npt.NDArray[np.float64],
#     X_cmd: npt.NDArray[np.float64],
#     X_ccd: npt.NDArray[np.float64],
#     prob: npt.NDArray[np.int64],
#     n_slices: int = 6,
#     save_path: Optional[Path] = None,
# ) -> None:
#     """Plot the phase-space according for slices of membership probability"""

#     # Make probability slices
#     xedges = np.linspace(0.0, 1.0, n_slices + 1)
#     corresponding_slice = np.digitize(prob, xedges)
#     unique_slices, slice_support = np.unique(corresponding_slice, return_counts=True)
#     slice_percentage = slice_support / len(prob) * 100

#     # Plot the data
#     n_rows = 4  # One for each parameter (pos, ast, cmd)
#     n_cols = n_slices  # number of columns
#     n_subplots = n_rows * n_cols  # total number of subplots

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

#     # Generate phase-space plots for each slice
#     for column_indx in range(n_cols):

#         # Data selection according to the current probability slice
#         slice_indx = column_indx + 1
#         indx_sel = np.where(corresponding_slice == slice_indx)

#         # Plot the positional data
#         pos_ax = axes[0, column_indx]
#         support_summary_ = support_summary(
#             edges=xedges,
#             supports=slice_support,
#             percentages=slice_percentage,
#             indx=slice_indx,
#         )
#         plot_pos_lb(
#             X=X_pos[indx_sel],
#             ax=pos_ax,
#             fig=fig,
#             title=support_summary_,
#         )
#         plot_sgr_stream_nom_position(ax=pos_ax)

#         # Plot the proper motions
#         ast_ax = axes[1, column_indx]
#         plot_proper_motions(X=X_ast[indx_sel], ax=ast_ax, fig=fig)

#         # Plot the CMD
#         cmd_ax = axes[2, column_indx]
#         plot_color_magnitude_diagram(X=X_cmd[indx_sel], ax=cmd_ax, fig=fig)

#         # Plot the CCD
#         cmd_ax = axes[3, column_indx]
#         plot_color_color_diagram(X=X_ccd[indx_sel], ax=cmd_ax, fig=fig)

#         if save_path is not None:
#             fig.savefig(save_path, dpi=300)


# def make_prob_slices_plots_cmd2(
#     X_pos: npt.NDArray[np.float64],
#     X_ast: npt.NDArray[np.float64],
#     X_cmd: npt.NDArray[np.float64],
#     X_ccd: npt.NDArray[np.float64],
#     prob: npt.NDArray[np.int64],
#     n_slices: int = 6,
#     save_path: Optional[Path] = None,
# ) -> None:
#     """Plot the phase-space according for slices of membership probability"""

#     # Make probability slices
#     xedges = np.linspace(0.0, 1.0, n_slices + 1)
#     corresponding_slice = np.digitize(prob, xedges)
#     unique_slices, slice_support = np.unique(corresponding_slice, return_counts=True)
#     slice_percentage = slice_support / len(prob) * 100

#     # Plot the data
#     n_rows = 4  # One for each parameter (pos, ast, cmd)
#     n_cols = n_slices  # number of columns
#     n_subplots = n_rows * n_cols  # total number of subplots

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

#     # Generate phase-space plots for each slice
#     for column_indx in range(n_cols):

#         # Data selection according to the current probability slice
#         slice_indx = column_indx + 1
#         indx_sel = np.where(corresponding_slice == slice_indx)

#         # Plot the positional data
#         pos_ax = axes[0, column_indx]
#         support_summary_ = support_summary(
#             edges=xedges,
#             supports=slice_support,
#             percentages=slice_percentage,
#             indx=slice_indx,
#         )
#         plot_pos_lb(
#             X=X_pos[indx_sel],
#             ax=pos_ax,
#             fig=fig,
#             title=support_summary_,
#         )
#         plot_sgr_stream_nom_position(ax=pos_ax)

#         # Plot the proper motions
#         ast_ax = axes[1, column_indx]
#         plot_proper_motions(X=X_ast[indx_sel], ax=ast_ax, fig=fig)

#         # Plot the CMD
#         cmd_ax = axes[2, column_indx]
#         plot_color_magnitude_diagram(X=X_cmd[indx_sel], ax=cmd_ax, fig=fig)

#         # Plot the CMD2
#         cmd_ax = axes[3, column_indx]
#         plot_color_magnitude_diagram(X=X_ccd[indx_sel], ax=cmd_ax, fig=fig)

#         if save_path is not None:
#             fig.savefig(save_path, dpi=300)


class PhaseSpaceType:
    POS = "pos"
    AST = "ast"
    CMD = "cmd"
    CCD = "ccd"


def make_prob_slices_plots_modular(
    labeled_data: Iterable[tuple[PhaseSpaceType, npt.NDArray[np.float64]]],
    prob: npt.NDArray[np.int64],
    n_slices: int = 6,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the phase-space according for slices of membership probability"""

    # Make probability slices
    xedges = np.linspace(0.0, 1.0, n_slices + 1)
    corresponding_slice = np.digitize(prob, xedges)
    unique_slices, slice_support = np.unique(corresponding_slice, return_counts=True)
    slice_percentage = slice_support / len(prob) * 100

    # Plot the data
    n_rows = len(labeled_data)  # One for each parameter
    n_cols = n_slices  # number of columns
    n_subplots = n_rows * n_cols  # total number of subplots

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Generate phase-space plots for each slice
    for column_indx in range(n_cols):

        # Data selection according to the current probability slice
        slice_indx = column_indx + 1
        indx_sel = np.where(corresponding_slice == slice_indx)

        for current_ax, (phase_space, X) in enumerate(labeled_data):

            if phase_space == PhaseSpaceType.POS:
                # Plot the positional data
                pos_ax = axes[current_ax, column_indx]
                support_summary_ = support_summary(
                    edges=xedges,
                    supports=slice_support,
                    percentages=slice_percentage,
                    indx=slice_indx,
                )
                plot_pos_lb(
                    X=X[indx_sel],
                    ax=pos_ax,
                    fig=fig,
                    title=support_summary_,
                )
                plot_sgr_stream_nom_position(ax=pos_ax)

            elif phase_space == PhaseSpaceType.AST:
                ast_ax = axes[current_ax, column_indx]
                plot_proper_motions(X=X[indx_sel], ax=ast_ax, fig=fig)

            elif phase_space == PhaseSpaceType.CMD:
                cmd_ax = axes[current_ax, column_indx]
                plot_color_magnitude_diagram(X=X[indx_sel], ax=cmd_ax, fig=fig)

            elif phase_space == PhaseSpaceType.CCD:
                cmd_ax = axes[3, column_indx]
                plot_color_color_diagram(X=X[indx_sel], ax=cmd_ax, fig=fig)

            else:
                raise ValueError(f"Invalid phase space type: {phase_space}")

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
