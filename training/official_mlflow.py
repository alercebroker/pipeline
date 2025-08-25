import os
import shutil
import argparse

def copy_mlflow_experiment(source_path: str, dest_base_path: str):
    # Validaciones iniciales
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"El path fuente no existe: {source_path}")
    
    experiment_id = os.path.basename(source_path.rstrip("/"))
    dest_path = os.path.join(dest_base_path, experiment_id)

    if os.path.exists(dest_path):
        raise FileExistsError(f"Ya existe un experimento con el mismo ID en destino: {dest_path}")
    
    print(f"Copiando experimento '{experiment_id}' desde:\n  {source_path}\na\n  {dest_path}")
    shutil.copytree(source_path, dest_path)
    print("✅ Copia completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copiar un experimento de MLflow a otro directorio.")
    parser.add_argument("--source", type=str, required=True, help="Path al experimento original (ej. mlruns/1)")
    parser.add_argument("--dest", type=str, required=True, help="Directorio donde se copiará el experimento")
    args = parser.parse_args()

    copy_mlflow_experiment(args.source, args.dest)