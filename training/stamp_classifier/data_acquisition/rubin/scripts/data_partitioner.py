import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

def partition_data(df, output_path, oid_col, candid_col, class_col,field=None, n_folds=5, random_state=42):
    """
    Partitions data into stratified train/val/test sets for k-fold CV and saves to a Parquet file.

    Parameters:
    - df: DataFrame
    - output_path: Path to save the resulting parquet file
    - oid_col: Column name for object ID
    - candid_col: Column name for unique source ID
    - class_col: Column name for class/label
    - n_folds: Number of folds for train/val (default 5)
    - random_state: Random seed

    Returns:
    - DataFrame with columns [oid_col, candid_col, class_col, partition]
    """

    print(df[class_col].value_counts())

    assert df[oid_col].is_unique, f"Duplicate '{oid_col}' values found in the input DataFrame"

    if field:
        train_val_df = df[df.target_name != field]
        test_df = df[df.target_name == field]#'Rubin_SV_095_-25']
        test_df['partition'] = 'test'
    else:
        # Step 1: Stratified test split (20%)
        train_val_df, test_df = train_test_split( #esto quiero hacerlo por target_name
            df,
            test_size=0.20,
            stratify=df[class_col],
            random_state=random_state)
        test_df['partition'] = 'test'

    # Step 2: Stratified K-Fold on remaining data
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    partitions = [test_df[[oid_col, candid_col, class_col, 'partition']]]

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df[class_col])):
        train_part = train_val_df.iloc[train_idx].copy()
        val_part = train_val_df.iloc[val_idx].copy()

        train_part['partition'] = f'training_{fold}'
        val_part['partition'] = f'validation_{fold}'

        partitions.extend([
            train_part[[oid_col, candid_col, class_col, 'partition']],
            val_part[[oid_col, candid_col, class_col, 'partition']]
        ])

    final_df = pd.concat(partitions, ignore_index=True)
    final_df.to_parquet(output_path, index=False)
    print(f"Partitioned data saved to: {output_path}")

    return final_df

if __name__ == "__main__":
    # Definición de columnas según el dataset actual
    oid_col = 'diaObjectId'
    candid_col = 'diaSourceId'
    class_col = 'class'
    field = None#'Rubin_SV_095_-25'

    # Cargar datos y crear carpeta de salida si no existe
    output_path = "./data/processed/partitions/ts_stamps_v0.0.4_dp1/partitions.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_parquet("./data/processed/ts_stamps_v0.0.4_dp1/objs.parquet")
    partitioned_df = partition_data(df, output_path, oid_col, candid_col, class_col,field)
