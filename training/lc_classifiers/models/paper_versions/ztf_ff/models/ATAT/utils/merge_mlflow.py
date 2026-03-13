import os
import shutil

# --- CONFIGURACIÓN ---

# 1. La carpeta donde pegaste los datos del OTRO servidor (Origen)
# Ajusta si tu carpeta se llama distinto.
SOURCE_MLRUNS = os.path.abspath("results/ml-runs-remoto")

# 2. Tu carpeta local donde quieres guardar todo (Destino)
TARGET_MLRUNS = os.path.abspath("results/ml-runs")

# 3. EL ID DEL EXPERIMENTO DESTINO
# IMPORTANTE: Revisa tu carpeta 'results/ml-runs'. ¿Qué número de carpeta ves ahí?
# En tu ejemplo de Target pusiste: '745273885923147472'
# En tu ls anterior tenías: '290561067201781048'
# PON AQUÍ EL ID DE LA CARPETA QUE YA EXISTE EN TU SERVIDOR ACTUAL:
TARGET_EXP_ID = "745273885923147472"  # <--- CAMBIA ESTO SI ES NECESARIO

def process_meta_yaml(run_path, new_exp_id, new_artifact_uri):
    """
    Lee el meta.yaml, actualiza rutas/IDs y agrega run_uuid si falta.
    """
    meta_path = os.path.join(run_path, "meta.yaml")
    
    if not os.path.exists(meta_path):
        return

    with open(meta_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    run_id_value = None
    has_run_uuid = False

    # Primera pasada: Leer valores y construir nuevas líneas
    for line in lines:
        stripped = line.strip()
        
        # Detectar el run_id original
        if stripped.startswith("run_id:"):
            run_id_value = stripped.split(":")[1].strip()
            new_lines.append(line) # Mantener run_id
            
        # Detectar si ya tiene run_uuid
        elif stripped.startswith("run_uuid:"):
            has_run_uuid = True
            new_lines.append(line)

        # Actualizar experiment_id
        elif stripped.startswith("experiment_id:"):
            new_lines.append(f"experiment_id: '{new_exp_id}'\n")
        
        # Actualizar artifact_uri con la nueva ruta absoluta
        elif stripped.startswith("artifact_uri:"):
            new_lines.append(f"artifact_uri: {new_artifact_uri}\n")
            
        else:
            # Mantener el resto (status, tiempos, etc.) tal cual
            new_lines.append(line)

    # Segunda parte: LA CORRECCIÓN CRÍTICA
    # Si encontramos un run_id pero NO un run_uuid, lo agregamos al final.
    if run_id_value and not has_run_uuid:
        print(f"   -> [FIX] Agregando 'run_uuid' faltante: {run_id_value}")
        new_lines.append(f"run_uuid: {run_id_value}\n")

    # Guardar cambios
    with open(meta_path, 'w') as f:
        f.writelines(new_lines)

def merge_runs():
    print(f"--- FUSIONANDO Y REPARANDO MLFLOW ---")
    print(f"Origen: {SOURCE_MLRUNS}")
    print(f"Destino: {TARGET_MLRUNS} (Exp ID: {TARGET_EXP_ID})")
    
    target_exp_path = os.path.join(TARGET_MLRUNS, TARGET_EXP_ID)
    if not os.path.exists(target_exp_path):
        print(f"ERROR: No encuentro la carpeta del experimento destino: {target_exp_path}")
        print("Por favor verifica la variable TARGET_EXP_ID en el script.")
        return

    # Buscar carpetas de experimentos en el origen
    # A veces el origen tiene carpeta '0', '1', o un ID largo. Iteramos sobre lo que haya.
    if not os.path.exists(SOURCE_MLRUNS):
         print(f"ERROR: No encuentro la carpeta origen: {SOURCE_MLRUNS}")
         return

    for exp_folder in os.listdir(SOURCE_MLRUNS):
        source_exp_path = os.path.join(SOURCE_MLRUNS, exp_folder)
        
        if not os.path.isdir(source_exp_path) or exp_folder == "models":
            continue

        print(f"\nLeyendo desde carpeta de experimento origen: {exp_folder}")

        # Iterar sobre las CORRIDAS (runs)
        for run_id in os.listdir(source_exp_path):
            source_run_path = os.path.join(source_exp_path, run_id)
            target_run_path = os.path.join(target_exp_path, run_id)
            
            # Verificar formato (longitud 32 chars tipica de run_id)
            if not os.path.isdir(source_run_path) or len(run_id) != 32:
                continue

            if os.path.exists(target_run_path):
                print(f"[OMITIDO] El Run {run_id} ya existe en destino.")
                continue

            # 1. MOVER
            try:
                shutil.move(source_run_path, target_run_path)
            except Exception as e:
                print(f"[ERROR] Moviendo {run_id}: {e}")
                continue

            # 2. REPARAR META.YAML (Rutas + UUID)
            new_artifact_uri = f"file://{os.path.join(target_run_path, 'artifacts')}"
            process_meta_yaml(target_run_path, TARGET_EXP_ID, new_artifact_uri)
            
            print(f"[EXITO] Run {run_id} movido y reparado.")

    print("\n--- PROCESO TERMINADO ---")

if __name__ == "__main__":
    # Verificación simple antes de correr
    print(f"Voy a mover runs desde '{SOURCE_MLRUNS}' hacia '{TARGET_MLRUNS}/{TARGET_EXP_ID}'")
    val = input("¿Confirmar? (escribe 'si'): ")
    if val.lower() == 'si':
        merge_runs()
    else:
        print("Cancelado.")