import itertools
import pandas as pd


def find_limits_rectangular_region(
    xmin: float, 
    xmax: float, 
    ymin: float, 
    ymax: float, 
    n_rows: int = 2, 
    n_columns: int = 3
) -> list[tuple[str, float, float, float, float]]:
    """Compute the subregions borders"""

    width = (xmax - xmin) / n_columns
    height = (ymax - ymin) / n_rows
    
    print(f"{width=}")
    print(f"{height=}")
    
    sub_rect = []
    
    for r, c in itertools.product(range(n_rows), range(n_columns)):
        
        xmin_sub = xmin + c * width
        xmax_sub = xmin_sub + width
        ymin_sub = ymin + r * height
        ymax_sub = ymin_sub + height
        
        sub_rect.append((f"{r}_{c}", xmin_sub, xmax_sub, ymin_sub, ymax_sub))
    
    return sub_rect


def create_grid(
    df: pd.DataFrame,
    n_rows: int = 2,
    n_columns: int = 4,
    col_l: str = "l",
    col_b: str = "b",
) -> dict[str, pd.DataFrame]:
    """Divide a dataframe into rectangular subregions."""

    lmin, lmax = df[col_l].min(), df[col_l].max()
    bmin, bmax = df[col_b].min(), df[col_b].max()

    # Compute subregions borders according to the number of cols and rows.
    subregions = find_limits_rectangular_region(lmin, lmax, bmin, bmax, n_rows, n_columns)

    output = {}
    for subreg in subregions:
        # filter
        id_str, lmin_s, lmax_s, bmin_s, bmax_s = subreg
        print(f"{lmin_s=:.2f} {lmax_s=:.2f} {bmin_s=:.2f} {bmax_s=:.2f}" )
        df_sub = df.query(f"{lmin_s} <= {col_l} <= {lmax_s} and {bmin_s} <= {col_b} <= {bmax_s}")
        output[id_str] = df_sub
    
    return output
