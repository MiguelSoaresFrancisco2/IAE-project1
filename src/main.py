from core.config import Config
from core.structs import (
    EMA_Variables,
    GeneralVariables,
    LTR_Variables,
    MF_Variables,
    MMR_Variables,
    PopularityVariables,
)

from core.utils import (
    plot_mmr_tradeoff,
    plot_training_history,
    compare_methods,
    print_examples_recommendations,
    save_logs,
    evaluate_method,
    setup_general_vars,
    setup_hyperparameters,
)

from personalization.ema import (
    get_session_recommendations,
    get_session_results,
    get_session_summary_df,
)
from rankers.pairwise_ltr import (
    build_pairwise_training_data,
    predict_pairwise_ltr,
    print_examples_pairwise_ltr,
    recommend_pairwise_ltr,
    train_pairwise_ltr,
)
from rankers.popularity import (
    evaluate_popularity,
    get_item_popularity,
    print_examples_popularity,
)
from rankers.mf_general import (
    predict_mf,
    print_rmse,
    recommend_mf,
)
from rankers.mf_sgd import train_mf_sgd
from rankers.mf_als import train_mf_als
from reranker.mmr import evaluate_mmr, print_mmr_example


if __name__ == "__main__":
    #######################################################################
    #                   CREATION AND SETUP OF VARIABLES                   #
    #######################################################################

    # Set up configuration and random seed
    config = Config()
    config.set_seed()

    # Initialize variable containers
    general_vars = GeneralVariables()
    popularity_vars = PopularityVariables()
    mf_sgd_vars = MF_Variables("mf_sgd")
    mf_als_vars = MF_Variables("mf_als")
    pairwise_ltr_vars = LTR_Variables("pairwise_ltr")

    # Data setup: load, split, evaluation caches, genre vectors, eligible users, etc.
    general_vars = setup_general_vars(config, general_vars)

    # Calculate item popularity
    popularity_vars.item_popularity = get_item_popularity(general_vars)

    # Example user for printing recommendations later
    example_user_id = general_vars.eligible_users[0] if general_vars.eligible_users else None

    # Set up hyperparameters for all methods
    setup_hyperparameters(config, mf_sgd_vars, mf_als_vars, pairwise_ltr_vars)

    # Store method variables in a dictionary for easier access during training and evaluation
    method_vars: dict[str, MF_Variables | LTR_Variables] = {
        "mf_sgd": mf_sgd_vars,
        "mf_als": mf_als_vars,
        "pairwise_ltr": pairwise_ltr_vars,
    }

    #
    #
    #######################################################################
    #                     POPULARITY-BASED RECOMMENDER                    #
    #######################################################################

    # Evaluate popularity-based recommender
    popularity_vars.results = evaluate_popularity(config, general_vars, popularity_vars)

    if config.PRINT_CONFIRM:
        print(popularity_vars.results.df[["recall@10", "ndcg@10", "diversity@10"]].mean())
        if example_user_id is not None:
            print_examples_popularity(config, general_vars, popularity_vars, example_user_id)

    save_logs(config, "popularity_eval", popularity_vars.results.rows, {"k": config.TOP_K})

    #
    #
    #######################################################################
    # MATRIX FACTORIZATION WITH SGD AND ALS AND PAIRWISE LTR RECOMMENDERS #
    #######################################################################

    for method in config.METHODS:
        print(f"Training {method.upper()}...")

        if method == "mf_sgd":
            method_vars[method].model = train_mf_sgd(config, general_vars)
        elif method == "mf_als":
            method_vars[method].model = train_mf_als(config, general_vars)
        elif method == "pairwise_ltr":
            error = False
            if config.LTR_MF_METHOD not in config.METHODS:
                print(
                    f"Error: LTR_MF_METHOD '{config.LTR_MF_METHOD}' must be in METHODS for pairwise LTR."
                )
                error = True
            elif config.LTR_MF_METHOD not in general_vars.done_methods_names:
                print(
                    f"Error: LTR_MF_METHOD '{config.LTR_MF_METHOD}' must be trained before pairwise LTR."
                )
                error = True
            else:
                pairwise_ltr_vars.mf_model = method_vars[config.LTR_MF_METHOD].model

            if error:
                print("Skipping pairwise LTR training due to configuration issues.")
                continue

            method_vars[method].train_data = build_pairwise_training_data(config, general_vars)
            if config.PRINT_CONFIRM:
                print_examples_pairwise_ltr(method_vars[method].train_data)

            pairwise_ltr_vars.model = train_pairwise_ltr(
                config,
                general_vars,
                popularity_vars,
                pairwise_ltr_vars,
                method_vars[config.LTR_MF_METHOD].model,
            )

        if config.SHOW_PLOTS or config.SAVE_IMAGES:
            y_label = "Loss" if method == "pairwise_ltr" else "Train RMSE"
            plot_training_history(
                config,
                method_vars[method].model["history"],
                title=f"MF-{method.upper()} Training RMSE"
                if method != "pairwise_ltr"
                else "Pairwise LTR Training Loss",
                xlabel="Epoch" if method != "mf_als" else "Iteration",
                ylabel=y_label,
                img_name=f"{method}_training_{'rmse' if method != 'pairwise_ltr' else 'loss'}.png",
            )

        method_vars[method].results = evaluate_method(
            config,
            general_vars,
            method_vars[method],
            recommend_func=recommend_mf if method != "pairwise_ltr" else recommend_pairwise_ltr,
            popularity_vars=popularity_vars if method == "pairwise_ltr" else None,
        )
        general_vars.done_methods_names.add(method)

        if config.PRINT_CONFIRM:
            print_examples_recommendations(
                config,
                general_vars,
                method_vars[method],
                recommend_func=recommend_mf if method != "pairwise_ltr" else recommend_pairwise_ltr,
                user_id=example_user_id if example_user_id is not None else 1,
                popularity_vars=popularity_vars if method == "pairwise_ltr" else None,
            )
            print(method_vars[method].results.df[["recall@10", "ndcg@10", "diversity@10"]].mean())
            if method != "pairwise_ltr":
                print_rmse(general_vars, method_vars[method].model)

        if config.COMPARE_METHODS:
            dfs_to_compare = [popularity_vars.results.df] + [
                method_vars[m].results.df for m in general_vars.done_methods_names
            ]
            names_to_compare = ["popularity"] + [m for m in general_vars.done_methods_names]
            compare_methods(dfs_to_compare, names_to_compare)

        save_logs(
            config,
            f"{method}_eval",
            method_vars[method].results.rows,
            method_vars[method].hyperparameters,
        )

    #
    #
    #######################################################################
    # MMR  #
    #######################################################################

    print("\nApplying MMR re-ranking...")
    mmr_vars = MMR_Variables()
    mmr_vars.hyperparameters = {
        "M": config.TOP_M,
        "alpha": config.MMR_ALPHA_VALUES,
        "base_ranker": "",  # placeholder, will be set during evaluation
    }

    if any(m not in general_vars.done_methods_names for m in config.METHODS_TO_APPLY_MMR):
        print("Error: All methods in METHODS_TO_APPLY_MMR must be trained before applying MMR.")
    else:
        for method in config.METHODS_TO_APPLY_MMR:
            print(f"\nApplying MMR to {method.upper()} recommendations...")

            if config.PRINT_CONFIRM:
                print_mmr_example(
                    config,
                    general_vars,
                    popularity_vars if method == "pairwise_ltr" else None,
                    method_vars[method],
                    user_id=example_user_id if example_user_id is not None else 1,
                )

            mmr_vars.results = evaluate_mmr(
                config,
                general_vars,
                method_vars[method],
                predict_func=predict_mf if method != "pairwise_ltr" else predict_pairwise_ltr,
                popularity_vars=popularity_vars if method == "pairwise_ltr" else None,
            )

            mmr_summary_df = (
                mmr_vars.results.df.groupby("alpha")[["recall@10", "ndcg@10", "diversity@10"]]
                .mean()
                .reset_index()
            )

            if config.PRINT_CONFIRM:
                print(mmr_summary_df)

            if config.SHOW_PLOTS or config.SAVE_IMAGES:
                plot_mmr_tradeoff(
                    config,
                    mmr_summary_df,
                    method,
                )

            mmr_vars.hyperparameters["base_ranker"] = method
            save_logs(
                config,
                f"mmr_{method}_eval",
                mmr_vars.results.rows,
                mmr_vars.hyperparameters,
            )

    #
    #
    #######################################################################
    # EMA  #
    #######################################################################

    ema_vars = EMA_Variables()
    if any(m not in general_vars.done_methods_names for m in config.EMA_MF_METHODS):
        print("Error: All methods in EMA_MF_METHODS must be trained before running EMA.")

    else:
        ema_vars.session_users = general_vars.eligible_users[:3]

        if config.PRINT_CONFIRM:
            print(ema_vars.session_users)

        for method in config.EMA_MF_METHODS:
            print(f"\nRunning EMA with {method.upper()} as base MF model...")

            ema_vars.mf_model = method_vars[method].model
            ema_vars.mf_method_name = method

            ema_vars.session_results = get_session_results(
                config,
                general_vars,
                popularity_vars,
                pairwise_ltr_vars,
                ema_vars,
            )

            ema_vars.session_summary_df = get_session_summary_df(
                config,
                ema_vars,
            )

            ema_vars.session_recommendations_df = get_session_recommendations(
                config,
                ema_vars,
            )

            save_logs(
                config,
                f"ema_{method}_session_eval",
                ema_vars.session_results,
                {
                    "mf_method": method,
                    "rho": config.EMA_RHO,
                    "beta": config.EMA_BETA,
                    "rounds": config.EMA_ROUNDS,
                    "k": config.TOP_K,
                },
                ema_logs=True,
            )
