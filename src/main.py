from core.config import Config
from core.structs import GeneralVariables, MF_Variables, PopularityVariables

from core.utils import (
    prepare_evaluation,
    plot_training_history,
    compare_methods,
    get_eligible_users,
    load_data,
    get_train_test_split,
    get_genre_vectors,
    save_logs,
)

from rankers.popularity import (
    evaluate_popularity,
    get_item_popularity,
    print_examples_popularity,
)
from rankers.mf_general import (
    prepare_md_data,
    get_md_data,
    evaluate_mf,
    print_examples_mf,
    print_rmse,
)
from rankers.mf_sgd import (
    train_mf_sgd,
)
from rankers.mf_als import (
    prepare_MF_ALS_DIMata,
    train_mf_als,
)


if __name__ == "__main__":
    config = Config()
    general_vars = GeneralVariables()
    config.set_seed()

    # Load data and prepare evaluation
    general_vars.ratings, general_vars.items = load_data(config)
    general_vars.train_df, general_vars.test_df = get_train_test_split(config, general_vars)
    (
        general_vars.all_users,
        general_vars.all_items,
        general_vars.train_items_by_user,
        general_vars.relevant_items_by_user,
    ) = prepare_evaluation(config, general_vars)

    # Get genre vectors for diversity calculation
    general_vars.item_genre_vectors = get_genre_vectors(general_vars)

    #
    #
    ##########################################################
    #              POPULARITY-BASED RECOMMENDER              #
    ##########################################################

    popularity_vars = PopularityVariables()

    # Calculate item popularity
    popularity_vars.item_popularity = get_item_popularity(general_vars)

    general_vars.eligible_users = get_eligible_users(config, general_vars)

    # Evaluate popularity-based recommender
    popularity_vars.results, popularity_vars.results_df = evaluate_popularity(
        config, general_vars, popularity_vars
    )

    if config.PRINT_CONFIRM:
        print(popularity_vars.results_df[["recall@10", "ndcg@10", "diversity@10"]].mean())
        print_examples_popularity(
            config,
            general_vars,
            popularity_vars,
            0,  # example user index
        )

    save_logs(config, "popularity_eval", popularity_vars.results, {"k": config.TOP_K})

    #
    #
    ##########################################################
    # MATRIX FACTORIZATION WITH SGD AND WITH ALS RECOMMENDER #
    ##########################################################

    mf_sgd_vars = MF_Variables("mf_sgd")
    mf_als_vars = MF_Variables("mf_als")
    mf_sgd_vars.hyperparameters = {
        "k": config.TOP_K,
        "d": config.MF_SGD_DIM,
        "lr": config.MF_SGD_LR,
        "reg": config.MF_SGD_REG,
        "epochs": config.MF_SGD_EPOCHS,
    }
    mf_als_vars.hyperparameters = {
        "k": config.TOP_K,
        "d": config.MF_ALS_DIM,
        "reg": config.MF_ALS_REG,
        "iters": config.MF_ALS_ITERS,
    }
    mf_vars = {
        "mf_sgd": mf_sgd_vars,
        "mf_als": mf_als_vars,
    }

    (
        general_vars.unique_user_ids,
        general_vars.unique_item_ids,
        general_vars.user_to_index,
        general_vars.item_to_index,
        general_vars.index_to_user,
        general_vars.index_to_item,
        general_vars.n_users,
        general_vars.n_items,
    ) = prepare_md_data(config, general_vars)

    general_vars.train_data, general_vars.test_data = get_md_data(config, general_vars)

    general_vars.user_ratings_train, general_vars.item_ratings_train = prepare_MF_ALS_DIMata(
        general_vars
    )

    for method in config.METHODS:
        print(f"Training {method.upper()}...")

        if method == "mf_sgd":
            mf_vars[method].model = train_mf_sgd(config, general_vars)
        elif method == "mf_als":
            mf_vars[method].model = train_mf_als(config, general_vars)

        plot_training_history(
            mf_vars[method].model["history"],
            title=f"MF-{method.upper()} Training RMSE",
            xlabel="Epoch" if method == "mf_sgd" else "Iteration",
            ylabel="Train RMSE",
            img_name=f"{method}_training_rmse.png",
            save_img=True,
        )

        mf_vars[method].results, mf_vars[method].results_df = evaluate_mf(
            config,
            general_vars,
            mf_vars[method],
        )
        general_vars.done_methods_names.add(method)

        if config.PRINT_CONFIRM:
            print_examples_mf(
                config,
                general_vars,
                mf_vars[method],
                1,  # example user index
            )
            print(mf_vars[method].results_df[["recall@10", "ndcg@10", "diversity@10"]].mean())
            print_rmse(general_vars, mf_vars[method].model)

        if config.COMPARE_METHODS:
            dfs_to_compare = [popularity_vars.results_df] + [
                mf_vars[m].results_df for m in general_vars.done_methods_names
            ]
            names_to_compare = [m for m in general_vars.done_methods_names]
            compare_methods(dfs_to_compare, names_to_compare)

        save_logs(
            config,
            f"{method}_eval",
            mf_vars[method].results,
            mf_vars[method].hyperparameters,
        )
