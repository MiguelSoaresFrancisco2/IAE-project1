import os
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
    compare_methods,
    save_logs,
    evaluate_method,
    setup_general_vars,
    setup_hyperparameters,
    train_method,
)

from personalization.ema import (
    get_session_recommendations,
    get_session_results,
    get_session_summary_df,
)
from rankers.pairwise_ltr import (
    build_pairwise_training_data,
    predict_pairwise_ltr,
)
from rankers.popularity import (
    evaluate_popularity,
    get_item_popularity,
    print_examples_popularity,
)
from rankers.mf_general import (
    predict_mf,
)
from reranker.mmr import evaluate_mmr, print_mmr_example


if __name__ == "__main__":
    #######################################################################
    #                   CREATION AND SETUP OF VARIABLES                   #
    #######################################################################
    
    print("==================================================")
    print("Setting up variables and data...")
    print("==================================================\n")

    # Set up configuration and random seed
    config = Config()
    config.set_seed()
    
    # Creating the logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

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

    # Building pairwise training data for LTR
    pairwise_ltr_vars.train_data = build_pairwise_training_data(config, general_vars)

    # Store method variables in a dictionary for easier access during training and evaluation
    methods_vars: dict[str, MF_Variables | LTR_Variables] = {
        "mf_sgd": mf_sgd_vars,
        "mf_als": mf_als_vars,
        "pairwise_ltr": pairwise_ltr_vars,
    }

    #
    #
    #######################################################################
    #                     POPULARITY-BASED RECOMMENDER                    #
    #######################################################################

    print("\n\n==================================================")
    print("Evaluating popularity-based recommender...")
    print("==================================================\n")

    # Evaluate popularity-based recommender
    popularity_vars.results = evaluate_popularity(config, general_vars, popularity_vars)

    # Print example recommendations for the example user
    if config.PRINT_CONFIRM and example_user_id is not None:
        print_examples_popularity(config, general_vars, popularity_vars, example_user_id)

    # Saving logs for popularity-based recommender evaluation in a JSONL file
    save_logs(
        config,
        general_vars,
        "popularity_eval",
        popularity_vars.results.rows,
        {"k": config.TOP_K},
    )

    #
    #
    #######################################################################
    # MATRIX FACTORIZATION WITH SGD AND ALS AND PAIRWISE LTR RECOMMENDERS #
    #######################################################################
    
    print("\n\n==================================================")
    print("Training and evaluating MF and LTR recommenders...")
    print("==================================================")

    # Train, evaluate, compare, and save logs for MF and LTR methods
    for method in config.METHODS:
        print(f"\n====== Training {method.upper()}... ======\n")

        # Train the method and store the model and training history
        methods_vars[method].model = train_method(
            config, general_vars, popularity_vars, methods_vars, method
        )
        if methods_vars[method].model is None:
            continue

        # Evaluate the method and store the results
        methods_vars[method].results = evaluate_method(
            config, general_vars, methods_vars[method], popularity_vars, example_user_id
        )

        # Mark this method as done for comparisons and MMR application
        general_vars.done_methods_names.add(method)

        # Compare methods (compare all methods trained so far)
        if config.COMPARE_METHODS:
            compare_methods(general_vars, popularity_vars, methods_vars)

        # Save logs for this method's evaluation in a JSONL file
        save_logs(
            config,
            general_vars,
            f"{method}_eval",
            methods_vars[method].results.rows,
            methods_vars[method].hyperparameters,
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
                    methods_vars[method],
                    user_id=example_user_id if example_user_id is not None else 1,
                )

            mmr_vars.results = evaluate_mmr(
                config,
                general_vars,
                methods_vars[method],
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
                general_vars,
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

            ema_vars.mf_model = methods_vars[method].model
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
                general_vars,
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
