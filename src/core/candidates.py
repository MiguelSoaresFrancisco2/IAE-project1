from core.structs import GeneralVariables


def get_candidates(general_vars: GeneralVariables, user_id: int) -> list:
    seen_items = general_vars.train_items_by_user.get(user_id, set())
    return list(general_vars.all_items - seen_items)