import logging
import numpy as np
import matplotlib.pyplot as plt

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
    df_train, train_data = get_arrays_std(FileConfig.TRAIN_DATA)
    vasiliev_data = get_arrays_vasiliev(FileConfig.VASILIEV_DATA)

    # Proper motion model computation
    ast_memb_s, ast_field = gen_ast(
        X_train=train_data["pm"], n_components=ParamsConfig.GMM_N_COMPONENTS
    )

    model = ModelContainer()
    model.modify_parameter(
        category="memb", space="pm", model=ast_memb_s, data=train_data["pm"]
    )
    model.modify_parameter(
        category="field", space="pm", model=ast_field, data=train_data["pm"]
    )
    model.run_expectation()

    # Positional model computation
    pos_memb = gen_pos_memb(X_train=vasiliev_data["pos_xy"])
    pos_field_s = gen_pos_field(
        X_train=train_data["pos_lb"], weights=model.prob_field, n_bins=10, n_intervals=2
    )

    model.modify_parameter(
        category="memb", space="pos", model=pos_memb, data=train_data["pos_xy"]
    )
    model.modify_parameter(
        category="field", space="pos", model=pos_field_s, data=train_data["pos_lb"]
    )
    model.run_expectation()

    # Photometric model computation
    phot_memb_s = gen_phot(X_train=train_data["cmd"], weights=model.prob_memb)
    phot_field_s = gen_phot(X_train=train_data["cmd"], weights=model.prob_field)

    model.modify_parameter(
        category="memb", space="phot", model=phot_memb_s, data=train_data["cmd"]
    )
    model.modify_parameter(
        category="field", space="phot", model=phot_field_s, data=train_data["cmd"]
    )
    model.run_expectation()

    # Main training loop
    for i in range(ParamsConfig.MAX_ITERATIONS):

        logging.info(f"Main Loop. Iteration {i}")

        # Maximization
        ast_memb_s = gen_ast_memb2(X_train=train_data["pm"], weights=model.prob_memb)
        pos_field_s = gen_pos_field(
            X_train=train_data["pos_lb"],
            weights=model.prob_field,
            n_bins=10,
            n_intervals=2,
        )
        phot_memb_s = gen_phot(X_train=train_data["cmd"], weights=model.prob_memb)
        phot_field_s = gen_phot(X_train=train_data["cmd"], weights=model.prob_field)

        model.modify_parameter(category="memb", space="pm", model=ast_memb_s)
        model.modify_parameter(category="field", space="pos", model=pos_field_s)
        model.modify_parameter(category="memb", space="phot", model=phot_memb_s)
        model.modify_parameter(category="field", space="phot", model=phot_field_s)
        model.run_expectation()

        # Check tolerance
        logging.info(f"eta = {model.eta}")
        if model.diff_eta < ParamsConfig.TOLERANCE:
            logging.info(f"Convergence achieved at iteration {i}")
            break

    # Plot results
    plt.hist(model.prob_memb, bins=20)
    plt.savefig(FileConfig.OUTPUT_PATH / "n_train_hist_prob.png")
    plt.close()
    
    make_prob_slices_plots_modular(
        labeled_data=(
            ("pos", train_data["pos_lb"]),
            ("ast", train_data["pm"]),
            ("cmd", train_data["cmd"]),
        ),
        prob=model.prob_memb,
        n_slices=5,
        save_path=FileConfig.OUTPUT_PATH / "n_train_histograms_modular.png",
    )

    # Save results
    df_train["prob_memb"] = model.prob_memb
    df_train["prob_field"] = model.prob_field


    # -------------------------------------------------------
    #
    #          Refine the model in a smaller region
    #
    # -------------------------------------------------------
    
    B_OFFSET = 3.5
    lmin = 5.107 - B_OFFSET
    lmax = 6.107 - B_OFFSET
    bmin = -14.587
    bmax = -13.587

    # df_train_small = df_train.query("l >= 1.607 and l <= 2.607 and b >= -14.587 and b <= -13.587")  # -3.5 deg
    # df_train_small = df_train.query("l >= 2.607 and l <= 3.607 and b >= -14.587 and b <= -13.587")   # -2.5 deg
    # df_train_small = df_train.query("l >= 3.607 and l <= 4.607 and b >= -14.587 and b <= -13.587")   # -1.5 deg
    df_train_small = df_train.copy().query("l >= @lmin and l <= @lmax and b >= @bmin and b <= @bmax")
    df_train_small.to_csv(FileConfig.OUTPUT_PATH / f"s_initial_{B_OFFSET:.1f}.csv", index=False)

    make_prob_slices_plots_modular(
        labeled_data=(
            ("pos", df_train_small[["l", "b"]].to_numpy()),
            ("ast", df_train_small[["pmra", "pmdec"]].to_numpy()),
            ("cmd", df_train_small[["mag_J-mag_Ks", "mag_Ks"]].to_numpy()),
        ),
        prob=df_train_small["prob_memb"].to_numpy(),
        n_slices=5,
        save_path=FileConfig.OUTPUT_PATH / "s_initial_train_histograms_modular.png",
    )

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
    model_s = ModelContainer(eta_0=model.eta)
    model_s.prob_memb = df_train_small["prob_memb"].to_numpy()
    model_s.prob_field = df_train_small["prob_field"].to_numpy()
    

    # PM
    ast_memb_s = gen_ast_memb2(X_train=train_data_small["pm"], weights=model_s.prob_memb)
    model_s.modify_parameter(
        category="memb", space="pm", model=ast_memb_s, data=train_data_small["pm"]
    )
    model_s.modify_parameter(
        category="field", space="pm", model=ast_field, data=train_data_small["pm"]
    )

    model_s.run_expectation()  # Added for run14

    # POS + CMD
    pos_field_s = gen_pos_field(
        X_train=train_data_small["pos_lb"],
        weights=model_s.prob_field,
        n_bins=4,
        n_intervals=2,
    )

    model_s.modify_parameter(
        category="memb", space="pos", model=pos_memb, data=train_data_small["pos_xy"]
    )
    model_s.modify_parameter(
        category="field", space="pos", model=pos_field_s, data=train_data_small["pos_lb"]
    )

    phot_memb_s = gen_phot(X_train=train_data_small["cmd"], weights=model_s.prob_memb)
    phot_field_s = gen_phot(X_train=train_data_small["cmd"], weights=model_s.prob_field)

    model_s.modify_parameter(
        category="memb", space="phot", model=phot_memb_s, data=train_data_small["cmd"]
    )
    model_s.modify_parameter(
        category="field", space="phot", model=phot_field_s, data=train_data_small["cmd"]
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
    df_train_small["prob_memb_s"] = model_s.prob_memb
    df_train_small["prob_field_s"] = model_s.prob_field
    df_train_small.to_csv(FileConfig.OUTPUT_PATH / f"s_results_{B_OFFSET:.1f}.csv", index=False)






