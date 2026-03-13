import mlflow
import shutil
import sys
import os
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# --- CONFIGURACIÓN ---
# Obtenemos la ruta absoluta de tu carpeta results/ml-runs
# Asumimos que corres el script desde la carpeta ATAT
tracking_path = os.path.abspath(os.path.join("results", "ml-runs"))

# Le decimos a MLflow que use esa ruta (agregando file:// para que sea URI válida)
mlflow.set_tracking_uri(f"file://{tracking_path}")

print(f"Usando base de datos de MLflow en: {tracking_path}")

client = MlflowClient()

def get_local_path(artifact_uri):
    """Convierte la URI de MLflow en una ruta de sistema operativo válida."""
    if "file://" in artifact_uri:
        # Quitamos el file:// y ajustamos para el SO
        path = artifact_uri.replace("file://", "")
        # En Windows a veces queda ///C:, ajustamos si es necesario
        if os.name == 'nt' and path.startswith('/'):
            path = path[1:]
        return os.path.dirname(path) # Retornamos la carpeta del run, no la de artifacts
    return None

def limpiar_carpetas_borradas():
    print("--- INICIANDO LIMPIEZA FÍSICA (Runs eliminados en UI) ---")
    experiments = client.search_experiments()
    count = 0
    
    for exp in experiments:
        # Buscar runs marcados como 'deleted'
        deleted_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=ViewType.DELETED_ONLY
        )
        
        for run in deleted_runs:
            run_id = run.info.run_id
            local_path = get_local_path(run.info.artifact_uri)
            
            if local_path and os.path.exists(local_path):
                try:
                    shutil.rmtree(local_path)
                    print(f"[BORRADO] Run {run_id} eliminado del disco.")
                    count += 1
                except Exception as e:
                    print(f"[ERROR] No se pudo borrar {run_id}: {e}")
            elif local_path:
                 print(f"[INFO] Run {run_id} ya no existía en disco.")
                 
    print(f"--- LIMPIEZA COMPLETADA: {count} carpetas eliminadas ---\n")

def reparar_status_padres():
    print("--- INICIANDO REPARACIÓN DE STATUS (De 4 a 3) ---")
    experiments = client.search_experiments()
    repaired_count = 0

    for exp in experiments:
        # 1. Buscamos TODOS los runs que estén FAILED (Status 4)
        failed_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FAILED'",
            run_view_type=ViewType.ACTIVE_ONLY
        )

        for parent in failed_runs:
            parent_id = parent.info.run_id
            
            # 2. Buscamos si este Run tiene HIJOS ACTIVOS
            # La etiqueta estándar para padre es 'mlflow.parentRunId'
            children = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'",
                run_view_type=ViewType.ACTIVE_ONLY
            )

            # Si no tiene hijos activos, quizás no es un padre o borraste todos los hijos.
            # Solo actuamos si tiene hijos y tú quieres "salvar" al padre.
            if len(children) > 0:
                # 3. Verificamos los hijos restantes
                # Si tienes hijos que TAMBIÉN fallaron y NO los borraste, no deberíamos poner el padre en verde.
                # Solo ponemos verde si TODOS los hijos ACTIVOS están FINISHED.
                
                all_children_finished = all(child.info.status == 'FINISHED' for child in children)
                
                if all_children_finished:
                    print(f"[REPARANDO] El padre {parent_id} estaba FAILED, pero sus hijos activos están FINISHED.")
                    client.set_terminated(run_id=parent_id, status="FINISHED")
                    repaired_count += 1
                else:
                    # Opcional: Mostrar por qué no se repara
                    failed_kids = [c.info.run_id for c in children if c.info.status == 'FAILED']
                    # print(f"[OMITIDO] Padre {parent_id} tiene hijos activos que aun fallan: {failed_kids}")

    print(f"--- REPARACIÓN COMPLETADA: {repaired_count} corridas padre actualizadas a FINISHED (3) ---")

if __name__ == "__main__":
    # Primero borramos la basura para que no interfiera en la lógica
    limpiar_carpetas_borradas()
    
    # Luego arreglamos los padres basándonos en los hijos que quedaron vivos
    reparar_status_padres()