import numpy as np
import pandas as pd

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix


class Sagittarius(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Sagittarius dwarf galaxy, as described in
        https://ui.adsabs.harvard.edu/abs/2003ApJ...599.1082M
    and further explained in
        https://www.stsci.edu/~dlaw/Sgr/.

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    Beta : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
    distance : `~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_Lambda_cosBeta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : `~astropy.units.Quantity`, optional, keyword-only
        The radial velocity of this object.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping("lon", "Lambda"),
            coord.RepresentationMapping("lat", "Beta"),
            coord.RepresentationMapping("distance", "distance"),
        ]
    }


SGR_PHI = (180 + 3.75) * u.degree  # Euler angles (from Law & Majewski 2010)
SGR_THETA = (90 - 13.46) * u.degree
SGR_PSI = (180 + 14.111534) * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
SGR_MATRIX = (
    np.diag([1.0, 1.0, -1.0])
    @ rotation_matrix(SGR_PSI, "z")
    @ rotation_matrix(SGR_THETA, "x")
    @ rotation_matrix(SGR_PHI, "z")
)


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, coord.Galactic, Sagittarius
)
def galactic_to_sgr():
    """Compute the Galactic spherical to heliocentric Sgr transformation matrix."""
    return SGR_MATRIX


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, Sagittarius, coord.Galactic
)
def sgr_to_galactic():
    """Compute the heliocentric Sgr to spherical Galactic transformation matrix."""
    return matrix_transpose(SGR_MATRIX)


sgr_stream_pos = coord.SkyCoord(
    Lambda=np.linspace(0, 2 * np.pi, 128) * u.radian,
    Beta=np.zeros(128) * u.radian,
    frame="sagittarius",
)

sgr_stream_gal = sgr_stream_pos.transform_to(coord.Galactic)

def compute_Sgr_coords(df: pd.DataFrame):
    """Compute Sgr X and Y coordinates using the previous transformation"""
    # df = df.copy()
    coords_lb = coord.SkyCoord(l=df["l"], b=df["b"], unit="deg", frame="galactic")
    coords_sgr = coords_lb.transform_to(Sagittarius)

    df["Sgr_lon"] = coords_sgr.Lambda.wrap_at(180 * u.deg).value
    df["Sgr_lat"] = coords_sgr.Beta.wrap_at(180 * u.deg).value
    return df

# plt.plot(
#     sgr_stream_gal.l.deg, sgr_stream_gal.b.deg, c="red", linestyle="dashed"
# )
