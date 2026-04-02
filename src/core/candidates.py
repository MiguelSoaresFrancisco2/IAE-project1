from core.structs import GeneralVariables


def get_candidates(general_vars: GeneralVariables, user_id: int) -> list:
    '''
    Returns a list of candidate item IDs for a given user. The candidates
    are defined as all items that the user has not interacted with in the
    training data. This function is essential for generating recommendations,
    as it ensures that we only consider items that are new to the user.
    '''
    
    # Get the set of items the user has interacted with in the training data
    seen_items = general_vars.train_items_by_user.get(user_id, set())
    
    # Returning the list of candidate items by taking the set difference between all items and seen items
    return list(general_vars.all_items - seen_items)