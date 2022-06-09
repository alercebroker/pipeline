import random


def create_random_string():
    letters = list("abcdefghijklmnopqrstuvwxyz")
    random.shuffle(letters)
    return ''.join(letters)
    

def create_data(n_data):
    data_batch = []
    for i in range(n_data):
        data_batch.append(
            {
                "id": random.getrandbits(32),
                "message": create_random_string(),
            }
        )
    return data_batch
