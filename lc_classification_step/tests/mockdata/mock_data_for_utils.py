import pandas as pd

messages_df = pd.DataFrame(
    [
        "aid1",
        "aid2",
        "aid3",
        "aid4",
        "aid5",
    ],
    index=[
        "oid1",
        "oid2",
        "oid3",
        "oid4",
        "oid5",
    ],
    columns=["aid"],
)
messages_df.index.name = "oid"

complete_classifications_df = pd.DataFrame(
    [
        [0.1, 0.2, 0.7],
        [0.3, 0.1, 0.6],
        [0.8, 0.1, 0.1],
        [0.2, 0.5, 0.3],
        [0.6, 0.2, 0.2],
    ],
    index=[
        "oid1",
        "oid2",
        "oid3",
        "oid4",
        "oid5",
    ],
    columns=[
        "class1",
        "class2",
        "class3",
    ],
)
complete_classifications_df.index.name = "oid"

incomplete_classifications_df = pd.DataFrame(
    [
        [0.1, 0.2, 0.7],
        [0.8, 0.1, 0.1],
        [0.6, 0.2, 0.2],
    ],
    index=[
        "oid1",
        "oid3",
        "oid5",
    ],
    columns=[
        "class1",
        "class2",
        "class3",
    ],
)
incomplete_classifications_df.index.name = "oid"
