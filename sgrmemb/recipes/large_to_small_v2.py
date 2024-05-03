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
from sgrmemb.loader import get_arrays_std, get_arrays_vasiliev
from sgrmemb.plots import make_prob_slices_plots_modular
from sgrmemb.settings import FileConfig, ParamsConfig
from sgrmemb.model import ModelContainer


def run():
    # Load data
    _, train_data = get_arrays_std(FileConfig.TRAIN_DATA)
    vasiliev_data = get_arrays_vasiliev(FileConfig.VASILIEV_DATA)

    model = ModelContainer()

    # Proper motion model computation
    ast_memb, ast_field = gen_ast(
        X_train=train_data["pm"], n_components=ParamsConfig.GMM_N_COMPONENTS
    )
    model.modify_parameters(
        categories=["memb", "field"],
        spaces=["pm", "pm"],
        models=[ast_memb, ast_field],
        data=[train_data["pm"], train_data["pm"]],
    )
    model.run_expectation()

    # Positional model computation
    pos_memb = gen_pos_memb(X_train=vasiliev_data["pos_xy"])
    pos_field = gen_pos_field(
        X_train=train_data["pos_lb"], weights=model.prob_field, n_bins=10, n_intervals=2
    )
    model.modify_parameters(
        categories=["memb", "field"],
        spaces=["pos", "pos"],
        models=[pos_memb, pos_field],
        data=[train_data["pos_xy"], train_data["pos_lb"]],
    )
    model.run_expectation()

    # Photometric model computation
    phot_memb = gen_phot(X_train=train_data["cmd"], weights=model.prob_memb)
    phot_field = gen_phot(X_train=train_data["cmd"], weights=model.prob_field)
    model.modify_parameters(
        categories=["memb", "field"],
        spaces=["phot", "phot"],
        models=[phot_memb, phot_field],
        data=[train_data["cmd"], train_data["cmd"]],
    )
    model.run_expectation()

    # Main training loop
    for i in range(ParamsConfig.MAX_ITERATIONS):

        logging.info(f"Main Loop. Iteration {i}")

        # Maximization
        ast_memb = gen_ast_memb2(X_train=train_data["pm"], weights=model.prob_memb)
        pos_field = gen_pos_field(
            X_train=train_data["pos_lb"],
            weights=model.prob_field,
            n_bins=10,
            n_intervals=2,
        )
        phot_memb = gen_phot(X_train=train_data["cmd"], weights=model.prob_memb)
        phot_field = gen_phot(X_train=train_data["cmd"], weights=model.prob_field)

        model.modify_parameters(
            categories=["memb", "field", "memb", "field"],
            spaces=["pm", "pos", "phot", "phot"],
            models=[ast_memb, pos_field, phot_memb, phot_field],
            data=[
                train_data["pm"],
                train_data["pos_lb"],
                train_data["cmd"],
                train_data["cmd"],
            ],
        )

        # Expectation
        model.run_expectation()

        # Check tolerance
        logging.info(f"eta = {model.eta}")
        if model.diff_eta < ParamsConfig.TOLERANCE:
            logging.info(f"Convergence achieved at iteration {i}")
            break


    # Results
    plt.hist(model.prob_memb, bins=50)
    plt.savefig(FileConfig.OUTPUT_PATH / "prob_hist.png")
    plt.close()
    make_prob_slices_plots_modular(
        labeled_data=(
            ("pos", train_data["pos_lb"]),
            ("ast", train_data["pm"]),
            ("cmd", train_data["cmd"]),
        ),
        prob=model.prob_memb,
        n_slices=10,
        save_path=FileConfig.OUTPUT_PATH / "histograms.png",
    )
    # Write results to file
    # df_output = pd.DataFrame.from_dict(model.get_data_as_dict())
    # df_output.to_csv(FileConfig.OUTPUT_PATH / "model.csv", index=False)
    t = Table(model.get_data_as_dict())
    t.write(FileConfig.OUTPUT_PATH / "results.fits", format="fits")
