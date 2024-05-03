import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sgrmemb.densitygen import (
    gen_ast,
    gen_ast_memb2,
    gen_pos_memb,
    gen_pos_field,
    gen_phot,
)
from sgrmemb.loader import get_arrays_std, get_arrays_vasiliev, get_subspace_data
from sgrmemb.plots import make_prob_slices_plots_modular
from sgrmemb.settings import FileConfig, ParamsConfig
from sgrmemb.model import ModelContainer


def run():
    """
    Compute membership probabilities using the standard procedure for AST, POS and PHOT parameter-space.
    """

    # Load data
    vasiliev_data = get_arrays_vasiliev(FileConfig.VASILIEV_DATA)

    

    # Small region ----------------------------
    df_train_small = pd.read_csv(FileConfig.DATA_PATH / "results/20230914/run07/train_probabilities.csv")
    df_train_small = df_train_small.query("l >= 1.607 and l <= 2.607 and b >= -14.587 and b <= -13.587")
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
    train_data_small = get_subspace_data(df_train_small, SPACE_PARAMS)
    #df_train_small.to_csv(FileConfig.OUTPUT_PATH / "s_train_data_small.csv", index=False)

    model_s = ModelContainer(eta_0=0.04)
    model_s.prob_memb = df_train_small.prob_memb.to_numpy()
    model_s.prob_field = df_train_small.prob_field.to_numpy()
    

    # PM
    ast_memb = gen_ast_memb2(X_train=train_data_small["pm"], weights=model_s.prob_memb)

    model_s.modify_parameter(
        category="memb", space="pm", model=ast_memb, data=train_data_small["pm"]
    )
    model_s.modify_parameter(
        category="field", space="pm", model=ast_field, data=train_data_small["pm"]  # HERE IS THE PROBLEM
    )

    # POS + CMD
    pos_memb = gen_pos_memb(X_train=vasiliev_data["pos_xy"])
    pos_field = gen_pos_field(
        X_train=train_data_small["pos_lb"],
        weights=model_s.prob_field,
        n_bins=4,
        n_intervals=1,
    )

    model_s.modify_parameter(
        category="memb", space="pos", model=pos_memb, data=train_data_small["pos_xy"]
    )
    model_s.modify_parameter(
        category="field", space="pos", model=pos_field, data=train_data_small["pos_lb"]
    )

    phot_memb = gen_phot(X_train=train_data_small["cmd"], weights=model_s.prob_memb)
    phot_field = gen_phot(X_train=train_data_small["cmd"], weights=model_s.prob_field)

    model_s.modify_parameter(
        category="memb", space="phot", model=phot_memb, data=train_data_small["cmd"]
    )
    model_s.modify_parameter(
        category="field", space="phot", model=phot_field, data=train_data_small["cmd"]
    )


    model_s.run_expectation()


    # Plot results
    plt.hist(model_s.prob_memb, bins=20)
    plt.savefig(FileConfig.OUTPUT_PATH / "s_train_hist_prob.png")
    plt.close()
    
    make_prob_slices_plots_modular(
        labeled_data=(
            ("pos", train_data["pos_lb"]),
            ("ast", train_data["pm"]),
            ("cmd", train_data["cmd"]),
        ),
        prob=model_s.prob_memb,
        n_slices=5,
        save_path=FileConfig.OUTPUT_PATH / "s_train_histograms_modular.png",
    )

    # Save results
    df_train_small["prob_memb"] = model_s.prob_memb
    df_train_small["prob_field"] = model_s.prob_field
    df_train_small.to_csv(FileConfig.OUTPUT_PATH / "s_train_probabilities.csv", index=False)






