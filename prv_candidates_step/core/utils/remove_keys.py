def remove_keys_from_dictionary(d: dict, keys_to_remove: list):
    """Returns a shallow copy without the keys in the keys_to_remove list"""
    return {k: d[k] for k in set(list(d.keys())) - set(keys_to_remove)}
