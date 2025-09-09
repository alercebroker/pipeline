import os
import shutil
import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

def _load_meta_from_parent(source: Path) -> dict:
    parent_meta = source.parent / "meta.yaml"
    if not parent_meta.exists():
        raise FileNotFoundError(f"No se encontró meta.yaml en el directorio padre: {parent_meta}")
    if yaml is None:
        raise RuntimeError("PyYAML no está instalado. Ejecuta: pip install pyyaml")
    with parent_meta.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _extract_id_extra_from_meta(meta: dict) -> str:
    exp_id = meta.get("experiment_id")
    if exp_id is None:
        raise ValueError("meta.yaml no contiene 'experiment_id'")
    # Normaliza por si viene con comillas
    exp_id = str(exp_id).strip().strip("'\"")
    if not exp_id:
        raise ValueError("El 'experiment_id' del meta.yaml está vacío")
    return exp_id

def copy_mlflow_experiment(source_path: str, dest_base_path: str):
    # Normalizar paths
    source = Path(source_path).resolve()
    dest_base = Path(dest_base_path).resolve()

    # Validaciones iniciales
    if not source.exists():
        raise FileNotFoundError(f"El path fuente no existe: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"El path fuente no es un directorio: {source}")

    # Cargar meta.yaml del padre y extraer id_extra desde experiment_id
    meta = _load_meta_from_parent(source)
    id_extra = _extract_id_extra_from_meta(meta)

    # Mantener el nombre del directorio source como experiment_dirname (igual que antes)
    experiment_dirname = source.name

    # Destinos (usa 'ml-runs' como en tu estructura)
    dest_id_extra_dir = dest_base / "ml-runs" / id_extra
    dest_experiment_dir = dest_id_extra_dir / experiment_dirname
    dest_meta = dest_id_extra_dir / "meta.yaml"

    # Validar conflictos
    if dest_experiment_dir.exists():
        raise FileExistsError(f"Ya existe un experimento con el mismo nombre en destino: {dest_experiment_dir}")
    if dest_meta.exists():
        raise FileExistsError(f"Ya existe un meta.yaml en {dest_meta}. Elimina o mueve ese archivo antes de copiar.")

    # Crear directorios destino necesarios
    dest_id_extra_dir.mkdir(parents=True, exist_ok=True)

    # Copiar meta.yaml del padre
    parent_meta = source.parent / "meta.yaml"
    print(f"[id_extra={id_extra}] Copiando meta.yaml:\n  {parent_meta}\n→ {dest_meta}")
    shutil.copy2(parent_meta, dest_meta)

    # Copiar el árbol del experimento
    print(f"Copiando experimento '{experiment_dirname}':\n  {source}\n→ {dest_experiment_dir}")
    shutil.copytree(source, dest_experiment_dir)

    print("✅ Copia completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copiar un experimento de MLflow a otro directorio. "
                    "El id_extra se toma de experiment_id en meta.yaml (del directorio padre de --source)."
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Path al experimento original (ej. /ruta/ml-runs/<experiment_dirname>)")
    parser.add_argument("--dest", type=str, required=True,
                        help="Directorio base destino (usará DEST/ml-runs/<id_extra>/)")
    args = parser.parse_args()

    copy_mlflow_experiment(args.source, args.dest)
