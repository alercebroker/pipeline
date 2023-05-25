import pandas as pd
import numpy


messages_df = pd.DataFrame(
    [
        "oid1",
        "oid2",
        "oid3",
        "oid4",
        "oid5",
    ],
    index=[
        "aid1",
        "aid2",
        "aid3",
        "aid4",
        "aid5",
    ],
    columns=["oid"],
)
messages_df.index.name = "aid"

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
    ],
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
    ],
)
incomplete_classifications_df.index.name = "aid"
