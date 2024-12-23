import pandas as pd
import numpy as np

def open_partitions(partitions):
    mapping_to_int = {
        key: idx for idx, key in enumerate(partitions['class_name'].unique())
    }

    def apply_mapping(label_str):
        return mapping_to_int[label_str]

    partitions["label_int"] = partitions.apply(
        lambda x: apply_mapping(x['class_name']), axis=1
    )
    return partitions, mapping_to_int

def ordered_partitions(df_lc, partitions, fold):
    this_partition = partitions[
        (partitions["partition"] == "training_%d" % fold)
        | (partitions["partition"] == "validation_%d" % fold)
        | (partitions["partition"] == "test")
    ]
    this_partition = this_partition.set_index("oid")
    this_partition_final = this_partition.filter(items=df_lc.index, axis=0)
    this_partition_final["unique_id"] = np.arange(len(this_partition_final))
    this_partition_final = this_partition_final.reset_index().rename(
        columns={"index": "oid"}
    )
    this_partition_final = this_partition_final.set_index("unique_id")
    return this_partition_final