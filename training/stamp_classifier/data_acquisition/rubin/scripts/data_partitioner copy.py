import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import argparse

# ==============================================================================
# === FUNCIÓN 1: CREAR PARTICIONES DESDE DATOS CRUDOS ===
# ==============================================================================
def partition_data(df, output_path, oid_col, class_col, n_folds=5, random_state=42, group_info_path=None, group_col='TNS_Name'):
    """
    Crea particiones (train/val/test) a partir de datos crudos, respetando grupos
    de objetos para evitar fugas de datos y las guarda en un archivo Parquet.
    """
    print("--- Iniciando proceso de particionamiento ---")
    print(f"Número de objetos iniciales: {df[oid_col].nunique()}")
    print("Distribución de clases inicial:")
    print(df[class_col].value_counts(normalize=True))

    # --- Manejo de Grupos (para evitar data leakage) ---
    if group_info_path:
        print(f"Cargando información de grupos desde: {group_info_path}")
        group_df = pd.read_csv(group_info_path)
        df = pd.merge(df, group_df[[oid_col, group_col]], on=oid_col, how='left')
        
        # Los objetos no presentes en el archivo de grupos se tratan como su propio grupo.
        df[group_col] = df[group_col].fillna(df[oid_col].astype(str))
        
        # Helper para hacer un split inicial que respete los grupos
        unique_groups = df.groupby(group_col)[class_col].first()
        train_groups, test_groups = train_test_split(
            unique_groups.index,
            test_size=0.20,
            stratify=unique_groups.values,
            random_state=random_state
        )
        train_val_df = df[df[group_col].isin(train_groups)].copy()
        test_df = df[df[group_col].isin(test_groups)].copy()
        test_df['partition'] = 'test'
        
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_generator = splitter.split(train_val_df, train_val_df[class_col], groups=train_val_df[group_col])
    else:
        # Particionamiento estratificado simple si no se provee información de grupos
        print("ADVERTENCIA: No se proporcionó archivo de grupos. Se procederá sin agrupar objetos.")
        train_val_df, test_df = train_test_split(
            df,
            test_size=0.20,
            stratify=df[class_col],
            random_state=random_state
        )
        test_df['partition'] = 'test'
        
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_generator = splitter.split(train_val_df, train_val_df[class_col], groups=train_val_df[oid_col])

    # --- Creación de los Folds ---
    partitions = [test_df.copy()]
    for fold, (train_idx, val_idx) in enumerate(split_generator):
        train_part = train_val_df.iloc[train_idx].copy()
        val_part = train_val_df.iloc[val_idx].copy()
        train_part['partition'] = f'training_{fold}'
        val_part['partition'] = f'validation_{fold}'
        partitions.extend([train_part, val_part])

    final_df = pd.concat(partitions, ignore_index=True)
    
    # Asegurar que las columnas necesarias están presentes
    cols_to_keep = [oid_col, class_col, 'partition']
    final_df = final_df[cols_to_keep]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"✔ Particiones de datos reales guardadas en: {output_path}")
    
    return final_df

# ==============================================================================
# === FUNCIÓN 2: AÑADIR DATOS SIMULADOS A PARTICIONES EXISTENTES ===
# ==============================================================================
def add_simulated_data(path_to_existing_partitions, simulated_df, output_path, oid_col, class_col):
    """
    Carga un archivo de particiones existente y añade datos simulados
    SOLO a los conjuntos de entrenamiento de cada fold.
    """
    print("\n--- Iniciando proceso de aumento de datos ---")
    print(f"Cargando particiones existentes desde: {path_to_existing_partitions}")
    partitions_df = pd.read_parquet(path_to_existing_partitions)
    
    training_folds = sorted([p for p in partitions_df['partition'].unique() if p.startswith('training')])
    if not training_folds:
        raise ValueError("No se encontraron particiones de entrenamiento ('training_...') en el archivo.")
        
    print(f"Se encontraron los siguientes folds de entrenamiento para aumentar: {training_folds}")
    
    simulated_to_add = []
    for fold in training_folds:
        sim_fold_df = simulated_df.copy()
        sim_fold_df['partition'] = fold
        simulated_to_add.append(sim_fold_df)
        
    all_simulated_df = pd.concat(simulated_to_add, ignore_index=True)
    
    # Combinar particiones originales con los nuevos datos simulados
    final_df = pd.concat([partitions_df, all_simulated_df[[oid_col, class_col, 'partition']]], ignore_index=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"✔ Particiones aumentadas con datos simulados guardadas en: {output_path}")
    return final_df


# ==============================================================================
# === BLOQUE PRINCIPAL DE EJECUCIÓN CON ARGUMENTOS ===
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Herramienta para crear y aumentar particiones de datos para Machine Learning.")
    parser.add_argument(
        'mode', 
        choices=['create-only', 'augment-only', 'create-and-augment'], 
        help="El modo de operación: \n"
             "'create-only': Solo crea particiones de datos reales.\n"
             "'augment-only': Solo añade datos simulados a particiones existentes.\n"
             "'create-and-augment': Crea particiones y luego las aumenta."
    )
    # --- Argumentos de Archivos ---
    parser.add_argument('--raw-data-path', help="Ruta al archivo Parquet de datos reales crudos (requerido para 'create').")
    parser.add_argument('--simulated-data-path', help="Ruta al archivo Parquet de datos simulados (requerido para 'augment').")
    parser.add_argument('--group-info-path', default=None, help="(Opcional) Ruta al archivo CSV con información de grupos de objetos.")
    parser.add_argument('--existing-partitions-path', help="Ruta al archivo de particiones existente (requerido para 'augment-only').")
    parser.add_argument('--output-path', required=True, help="Ruta del archivo de salida final.")
    
    # --- Argumentos de Configuración ---
    parser.add_argument('--oid-col', default='oid', help="Nombre de la columna de ID de objeto.")
    parser.add_argument('--class-col', default='class', help="Nombre de la columna de clase/etiqueta.")

    args = parser.parse_args()

    # --- Lógica según el modo seleccionado ---
    
    if args.mode == 'create-only':
        if not args.raw_data_path:
            parser.error("'create-only' mode requires --raw-data-path.")
        
        print(f"MODO: Crear particiones únicamente.")
        df_real = pd.read_parquet(args.raw_data_path)
        partition_data(
            df=df_real,
            output_path=args.output_path,
            oid_col=args.oid_col,
            class_col=args.class_col,
            group_info_path=args.group_info_path
        )

    elif args.mode == 'augment-only':
        if not args.existing_partitions_path or not args.simulated_data_path:
            parser.error("'augment-only' mode requires --existing-partitions-path and --simulated-data-path.")
            
        print(f"MODO: Aumentar particiones existentes.")
        df_simulated = pd.read_parquet(args.simulated_data_path)
        add_simulated_data(
            path_to_existing_partitions=args.existing_partitions_path,
            simulated_df=df_simulated,
            output_path=args.output_path,
            oid_col=args.oid_col,
            class_col=args.class_col
        )

    elif args.mode == 'create-and-augment':
        if not args.raw_data_path or not args.simulated_data_path:
            parser.error("'create-and-augment' mode requires --raw-data-path and --simulated-data-path.")

        print(f"MODO: Crear y luego aumentar particiones.")
        
        # Paso 1: Crear particiones con datos reales
        df_real = pd.read_parquet(args.raw_data_path)
        # Definir una ruta intermedia para las particiones de solo reales
        real_partitions_intermediate_path = args.output_path.replace('.parquet', '_real_only.parquet')
        
        partition_data(
            df=df_real,
            output_path=real_partitions_intermediate_path,
            oid_col=args.oid_col,
            class_col=args.class_col,
            group_info_path=args.group_info_path
        )
        
        # Paso 2: Aumentar las particiones recién creadas
        df_simulated = pd.read_parquet(args.simulated_data_path)
        add_simulated_data(
            path_to_existing_partitions=real_partitions_intermediate_path,
            simulated_df=df_simulated,
            output_path=args.output_path, # Guardar el resultado final en la ruta de salida principal
            oid_col=args.oid_col,
            class_col=args.class_col
        )
        
        print(f"\nProceso completo. El archivo final se encuentra en: {args.output_path}")
        print(f"(El archivo intermedio de solo reales se guardó en: {real_partitions_intermediate_path})")

    print("\n¡Proceso finalizado con éxito!")