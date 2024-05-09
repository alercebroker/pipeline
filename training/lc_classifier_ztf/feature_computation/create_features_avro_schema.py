import pandas as pd


features = pd.read_parquet("data_231206_ao_features/consolidated_features.parquet")
assert features.columns[-1] == "shorten"
feature_list = features.columns[:-1].to_list()
feature_list.sort()

print(
    """
{
\t"name": "features_record_ztf",
\t"type": "record",
\t"fields": ["""
)

for i, feature in enumerate(feature_list):
    splitted_feature = feature.split("_")
    feature_root = "_".join(splitted_feature[:-1]).replace("-", "_")
    feature_ending = splitted_feature[-1]
    if feature_ending == "g":
        feature_root += "_1"
    elif feature_ending == "r":
        feature_root += "_2"
    elif feature_ending == "g,r":
        feature_root += "_12"

    if i == (len(feature_list) - 1):
        print(f'\t\t{{"name": "{feature_root}", "type": ["float", "null"]}}')
    else:
        print(f'\t\t{{"name": "{feature_root}", "type": ["float", "null"]}},')

print(
    """\t]
}
"""
)
