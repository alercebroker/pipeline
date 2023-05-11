import pandas as pd
import numpy


messages_df = pd.DataFrame(
    [
        {"aid": "aid1"},
        {"aid": "aid2"},
        {"aid": "aid3"},
        {"aid": "aid4"},
        {"aid": "aid5"},
    ]
)

complete_classifications_df = pd.DataFrame(
    [
        [0.1, 0.2, 0.7],
        [0.3, 0.1, 0.6],
        [0.8, 0.1, 0.1],
        [0.2, 0.5, 0.3],
        [0.6, 0.2, 0.2],

    ],
    index=[
        "aid1",
        "aid2",
        "aid3",
        "aid4",
        "aid5",
    ],
    columns=[
       "class1",
       "class2",
       "class3",
    ]
)
complete_classifications_df.index.name = "aid"

incomplete_classifications_df = pd.DataFrame(
    [
        [0.1, 0.2, 0.7],
        [0.8, 0.1, 0.1],
        [0.6, 0.2, 0.2],

    ],
    index=[
        "aid1",
        "aid3",
        "aid5",
    ],
    columns=[
       "class1",
       "class2",
       "class3",
    ]
)
incomplete_classifications_df.index.name = "aid"
