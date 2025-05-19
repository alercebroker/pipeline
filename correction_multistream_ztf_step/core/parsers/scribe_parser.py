
import json

def dict_splitter(correction_dict):

    base_dict = correction_dict.copy()
    
    base_dict.pop('payload')
    list_new_dicts = []

    for d in correction_dict['payload']: ## confirmar si sigue siendo payolad

        aux_dict = base_dict.copy()
        aux_dict['payload'] = d
        list_new_dicts.append(aux_dict)

    return list_new_dicts

with open('parsers_utils/test_correction.json') as f:
    d = json.load(f)
    


