from prv_candidates_step.core.utils.remove_keys import remove_keys_from_dictionary

def test_remove_keys():
    test_dict = {
        'a': 'a',
        'b': 'b',
        'c': 'c',
        'hehe': 'hehe'
    }
    new_dict = remove_keys_from_dictionary(test_dict, ['b', 'hehe'])
    assert set(new_dict.keys()) == set(['a', 'c'])