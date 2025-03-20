import pandas as pd
from scripts.data_partitioner import create_partitions

def process_dataset(input_path: str, output_path: str, class_mapping: dict) -> pd.DataFrame:
    """Load, map, and filter the dataset according to predefined class mappings."""
    
    objects = pd.read_parquet(input_path)
    objects["alerceclass"] = objects["alerceclass"].map(class_mapping)
    objects = objects.dropna(subset=["alerceclass"])
    objects.to_parquet(output_path)

    return objects


def partition_dataset(objects: pd.DataFrame, partition_config: dict, num_folds: int = 5):
    """Create dataset partitions based on the given configuration."""
    create_partitions(
        objects=objects,
        dir_save_partition=partition_config["name_new_partition"],
        num_folds=num_folds,
        dict_keep_test=partition_config
    )


if __name__ == "__main__":
    # Define file paths
    input_file = "raw/objects.parquet"
    output_file = "raw/objects_sanchez_tax_with_others.parquet"

    # Define class mapping
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

    # Process dataset
    objects = process_dataset(input_file, output_file, class_mapping)

    # Partitioning configuration
    ROOT = f"preprocessed/partitions"
    partition_config = {
        "name_prev_partition": f"{ROOT}/241209_ndetge8",
        "name_new_partition": f"{ROOT}/241209_ndetge8_sanchez_tax_with_others",
    }

    num_folds = 5

    # Create partitions
    partition_dataset(objects, partition_config, num_folds)
