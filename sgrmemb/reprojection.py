import pandas as pd
import numpy as np

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix


R = np.array(
    [
        [0.20482281, -0.83729492, -0.50693671],
        [0.83729492, 0.41811959, -0.35229706],
        [0.50693671, -0.35229706, 0.78670322],
    ]
)

Sgr_dSph_PA = (8.5) * u.deg  # PA angle derived in notebook deprojection/ellipse.ipynb
SGR_ROT_MATRIX = rotation_matrix(Sgr_dSph_PA, "x") @ R


class SgrMajorMinorCoordFrame(coord.BaseCoordinateFrame):

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping("lon", "lon"),
            coord.RepresentationMapping("lat", "lat"),
            coord.RepresentationMapping("distance", "distance"),
        ]
    }


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, coord.ICRS, SgrMajorMinorCoordFrame
)
def galactic_to_sgr():
    """Compute the Galactic spherical to heliocentric Sgr transformation matrix."""
    return SGR_ROT_MATRIX


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, SgrMajorMinorCoordFrame, coord.ICRS
)
def sgr_to_galactic():
    """Compute the heliocentric Sgr to spherical Galactic transformation matrix."""
    return matrix_transpose(SGR_ROT_MATRIX)


def compute_Sgr_XY(df: pd.DataFrame):
    """Compute Sgr X and Y coordinates using the previous transformation"""
    df = df.copy()
    coords_radec = coord.SkyCoord(ra=df["ra"], dec=df["dec"], unit="deg", frame="icrs")
    coords_sgr = coords_radec.transform_to(SgrMajorMinorCoordFrame)

    df["SgrX"] = coords_sgr.lon.wrap_at(180 * u.deg).value
    df["SgrY"] = coords_sgr.lat.wrap_at(180 * u.deg).value
    return df
