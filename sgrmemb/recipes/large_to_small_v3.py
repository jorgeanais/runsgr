import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table

from sgrmemb.densitygen import (
    gen_ast,
    gen_ast_memb2,
    gen_pos_memb,
    gen_pos_field,
    gen_phot,
)
from sgrmemb.loader import get_subspace_data, SPACE_PARAMS
from sgrmemb.plots import make_prob_slices_plots_modular
from sgrmemb.settings import FileConfig, ParamsConfig
from sgrmemb.model import ModelContainer
from sgrmemb.recipes import basic
from sgrmemb.cutting_tools import create_grid


def fine_tune(
    large_scale_model: ModelContainer,
    df: pd.DataFrame,
    label: str = ""
) -> ModelContainer:
    """
    Fine tune the model for a given region.
    
    Parameters
    ----------
    large_scale_model : ModelContainer
        The model at large scale.

    df : pd.DataFrame
        The dataframe with the data for the region.

    label : str
        An label for identifying the region.
    """

    logging.info(f"Perfoming fine tunning {label}")

    # Create the train data structure from the dataframe
    train_data = get_subspace_data(df, SPACE_PARAMS)

    # Make a copy of the large scale model and modify only the data content
    model = copy.deepcopy(large_scale_model)

    model.modify_parameters(
        categories=["memb", "field"],
        spaces=["pm", "pm"],
        models=[None, None],
        data=[train_data["pm"], train_data["pm"]],
    )
    model.modify_parameters(
        categories=["memb", "field"],
        spaces=["pos", "pos"],
        models=[None, None],
        data=[train_data["pos_xy"], train_data["pos_lb"]],
    )
    model.modify_parameters(
        categories=["memb", "field"],
        spaces=["phot", "phot"],
        models=[None, None],
        data=[train_data["cmd"], train_data["cmd"]],
    )

    # Update the probabilities (correct shape)
    model.prob_memb = df["prob_memb"].to_numpy()
    model.prob_field = df["prob_field"].to_numpy()
    
    # Update the parameters iteratively only for PM (memb) and PHOT (memb, field)
    for i in range(ParamsConfig.MAX_ITERATIONS):
        
        # Maximization
        ast_memb = gen_ast_memb2(X_train=train_data["pm"], weights=model.prob_memb)
        phot_memb = gen_phot(X_train=train_data["cmd"], weights=model.prob_memb)
        phot_field = gen_phot(X_train=train_data["cmd"], weights=model.prob_field)

        model.modify_parameters(
            spaces=["pm", "phot", "phot"],
            categories=["memb", "memb", "field"],
            models=[ast_memb, phot_memb, phot_field],
            data=[
                train_data["pm"],
                train_data["cmd"],
                train_data["cmd"],
            ],
        )

        # Expetation
        model.run_expectation()

        # Check tolerance
        print(f"eta = {model.eta}")
        if model.diff_eta < ParamsConfig.TOLERANCE:
            print(f"Convergence achieved at iteration {i}")
            break
    

    return model



def run() -> tuple[ModelContainer, pd.DataFrame]:

    # Compute the model at large scale
    model_ls, df_ls = basic.run()

    # Add the computed probabilities and parameters
    df_ls_p = pd.concat(
        [
            df_ls.reset_index(),
            pd.DataFrame(model_ls.get_data_as_dict())
        ], axis=1
    )

    # Create the smaller regions
    regions = create_grid(df_ls_p, n_rows=2, n_columns=4)

    # Apply fine tunning to each region
    for region_id, df_region in regions.items():

        logging.info(f"Fine tune of {region_id=}")
        finetuned_model = fine_tune(model_ls, df_region)

        break

    pass
