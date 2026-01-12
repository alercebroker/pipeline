import os
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Asegurar que el root del repo esté en sys.path para importar 'feature_step.utils'
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from feature_step.utils.feature_io import find_feature_files, load_features_series

# Path base donde se guardan los CSVs
csvs_folder = str(REPO_ROOT / "feature_step" / "csvs")

# Buscar recursivamente todos los archivos de features
feature_files = find_feature_files(csvs_folder)

# Construir un DataFrame con filas = objetos y columnas = features
rows_dict = {}
for fpath in feature_files:
    series = load_features_series(fpath)
    if series is None or len(series) == 0:
        continue
    # index amigable: <batchId>__<oid>
    batch_id = os.path.basename(os.path.dirname(fpath))
    base = os.path.basename(fpath)
    obj_id = base[:-len("_features.csv")] if base.endswith("_features.csv") else os.path.splitext(base)[0]
    row_key = f"{batch_id}__{obj_id}"
    rows_dict[row_key] = series

if not rows_dict:
    print("No se encontraron CSVs de features para procesar.")
    raise SystemExit(0)

all_features = pd.DataFrame.from_dict(rows_dict, orient="index")
print(all_features.shape)
# Plot the distribution of each feature
output_folder = str(REPO_ROOT / "feature_distributions")
os.makedirs(output_folder, exist_ok=True)

for feature in all_features.columns:
    feature_folder = os.path.join(output_folder, feature)
    os.makedirs(feature_folder, exist_ok=True)

    # Convertir siempre a numérico y ploteo si hay datos válidos
    col_num = pd.to_numeric(all_features[feature], errors="coerce").dropna()
    if col_num.empty:
        continue

    unique_vals = col_num.unique()
    plt.figure()
    if len(unique_vals) == 1:
        v = float(unique_vals[0])
        # Histograma para valor constante (sin KDE, con rango estrecho)
        plt.hist(col_num, bins=1, range=(v - 1e-9, v + 1e-9))
    else:
        sns.histplot(col_num, bins=50, kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(feature_folder, f"{feature}_distribution.png"))
    plt.close()

print(f"Feature-specific distributions saved to individual folders in {output_folder}")