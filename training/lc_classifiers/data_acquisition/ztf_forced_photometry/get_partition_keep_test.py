import pandas as pd
from scripts.data_partitioner import create_partitions
from sklearn.model_selection import StratifiedShuffleSplit

class_mapping = {
    "SNIa": "SNIa",
    "SNIbc": "SNIbc",
    "SNIIb": 'Others',  # Ignored class
    "SNII": "SNII",
    "SNIIn": "SNII",
    "SLSN": "SLSN",
    "TDE": 'Others',  # Ignored class
    "Microlensing": 'Others',  # Ignored class
    "QSO": "QSO",
    "AGN": "AGN",
    "Blazar": "Blazar",
    "YSO": "YSO",
    "CV/Nova": "CV/Nova",
    "LPV": "LPV",
    "EA": "E",
    "EB/EW": "E",
    "Periodic-Other": "Periodic-Other",
    "RSCVn": "Periodic-Other",
    "CEP": "CEP",
    "RRLab": "RRL",
    "RRLc": "RRL",
    "DSCT": "DSCT",
}

def process_dataset(input_path: str, output_path: str, class_mapping: dict) -> pd.DataFrame:
    """Load, map, and filter the dataset according to predefined class mappings."""
    
    objects = pd.read_parquet(input_path)
    objects["alerceclass"] = objects["alerceclass"].map(class_mapping)
    objects = objects.dropna(subset=["alerceclass"])
    objects.to_parquet(output_path)

    return objects

def partition_dataset(objects: pd.DataFrame, partition_config: dict, num_folds: int = 5, SplitObject = None):
    """Create dataset partitions based on the given configuration."""
    create_partitions(
        objects=objects,
        dir_save_partition=partition_config["name_new_partition"],
        num_folds=num_folds,
        dict_keep_test=partition_config,
        SplitObject=SplitObject, 
    )


if __name__ == "__main__":
    taxonomia = 'sanchez'
    ROOT = f"preprocessed/partitions"
    input_file = "raw/objects.parquet"

    num_folds = 20
    SplitObject = StratifiedShuffleSplit

    if taxonomia == 'sanchez':
        output_file = "raw/objects_sanchez_tax.parquet"
        name_new_partition = f"{ROOT}/241209_ndetge8_sanchez_tax_{num_folds}folds"
        objects = process_dataset(input_file, output_file, class_mapping)
    else:
        objects = pd.read_parquet("raw/objects.parquet")
        name_new_partition = f"{ROOT}/241209_ndetge8_{num_folds}folds"

    # Partitioning configuration
    partition_config = {
        "name_prev_partition": f"{ROOT}/241209_ndetge8",
        "name_new_partition": name_new_partition,
    }   

    # Create partitions
    partition_dataset(objects, partition_config, num_folds, SplitObject)
