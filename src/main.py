from core.config import Config
from core.structs import GeneralVariables, MF_SGDVariables, PopularityVariables

from core.utils import (
    prepare_evaluation,
    plot_training_history,
    compare_methods,
    print_rmse,
    get_eligible_users,
    load_data,
    get_train_test_split,
    get_genre_vectors,
    save_logs,
)

from rankers.popularity import evaluate_popularity, get_item_popularity, print_examples_popularity
from rankers.mf_sgd import (
    evaluate_mf_sgd,
    predict_mf_sgd,
    prepare_md_sgd_data,
    train_mf_sgd,
    get_md_sgd_data,
    print_examples_mf_sgd,
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
    #   MATRIX FACTORIZATION WITH SGD RECOMMENDER (MF_SGD)   #
    ##########################################################

    mf_sgd_vars = MF_SGDVariables()

    (
        general_vars.unique_user_ids,
        general_vars.unique_item_ids,
        general_vars.user_to_index,
        general_vars.item_to_index,
        general_vars.index_to_user,
        general_vars.index_to_item,
        general_vars.n_users,
        general_vars.n_items,
    ) = prepare_md_sgd_data(config, general_vars)

    general_vars.train_data, general_vars.test_data = get_md_sgd_data(config, general_vars)

    mf_sgd_vars.model = train_mf_sgd(config, general_vars)

    plot_training_history(
        mf_sgd_vars.model["history"],
        title="MF-SGD Training RMSE",
        xlabel="Epoch",
        ylabel="Train RMSE",
        img_name="mf_sgd_training_rmse.png",
        save_img=True,
    )

    mf_sgd_vars.results, mf_sgd_vars.results_df = evaluate_mf_sgd(
        config,
        general_vars,
        mf_sgd_vars,
    )

    if config.PRINT_CONFIRM:
        print(mf_sgd_vars.results_df[["recall@10", "ndcg@10", "diversity@10"]].mean())
        print_examples_mf_sgd(
            config,
            general_vars,
            mf_sgd_vars,
            1,  # example user index
        )

    if config.COMPARE_METHODS:
        compare_methods(
            [popularity_vars.results_df, mf_sgd_vars.results_df], ["popularity", "mf_sgd"]
        )

    save_logs(
        config,
        "mf_sgd_eval",
        mf_sgd_vars.results,
        {
            "k": config.TOP_K,
            "d": mf_sgd_vars.model["params"]["d"],
            "lr": mf_sgd_vars.model["params"]["lr"],
            "reg": mf_sgd_vars.model["params"]["reg"],
            "epochs": mf_sgd_vars.model["params"]["epochs"],
        },
    )

    if config.PRINT_CONFIRM:
        print_rmse(general_vars, mf_sgd_vars.model, predict_mf_sgd)


    #
    #
    ##########################################################
    #   MATRIX FACTORIZATION WITH SGD RECOMMENDER (MF_SGD)   #
    ##########################################################


