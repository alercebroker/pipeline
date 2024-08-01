import pandas as pd
import numpy as np
from tqdm import tqdm

import copy
import glob

MAX_DAYS = 2072
num_obs_before_det = 10


def find_nearest_unique_times(lc_mjds, grid_mjds):
    selected_indices = []
    used_indices = set()

    for mjd in grid_mjds:
        differences = np.abs(lc_mjds - mjd)
        index = np.argmin(differences)

        while index in used_indices:
            differences[index] = np.inf
            index = np.argmin(differences)
            if np.isinf(differences[index]):
                return selected_indices

        selected_indices.append(index)
        used_indices.add(index)

    return np.array(selected_indices)


def pad_list(lc, nepochs, dict_info, aux_times, flag_detections):
    if dict_info["type_windows"] == "windows":
        pad_num = dict_info["max_obs"] - nepochs
        if pad_num >= 0:
            return np.pad(lc, (0, pad_num), "constant", constant_values=(0, 0))

    elif dict_info["type_windows"] == "linspace_idx":
        pad_num = dict_info["max_obs"] - nepochs
        if pad_num >= 0:
            return np.pad(lc, (0, pad_num), "constant", constant_values=(0, 0))
        else:
            return np.array(lc)[
                np.linspace(0, nepochs - 1, num=dict_info["max_obs"]).astype(int)
            ]

    elif dict_info["type_windows"] == "logspace_idx":
        indices = (
            np.logspace(
                start=np.log10(1),
                stop=np.log10(nepochs),
                num=dict_info["max_obs"],
                endpoint=True,
                base=10,
            )
            - 1
        )
        indices = np.unique(np.round(indices).astype(int))
        indices = np.clip(indices, 0, nepochs - 1)
        lc_filtered = np.array(lc)[indices]
        pad_num = dict_info["max_obs"] - len(lc_filtered)
        if pad_num >= 0:
            lc_padded = np.pad(
                lc_filtered, (0, pad_num), "constant", constant_values=(0, 0)
            )
        return lc_padded

    elif dict_info["type_windows"] == "logspace_times":
        grid_mjds = np.geomspace(1, MAX_DAYS, num=dict_info["max_obs"]) + np.min(
            aux_times
        )
        logspace_mjd_idxs = find_nearest_unique_times(aux_times, grid_mjds)
        lc_filtered = lc[logspace_mjd_idxs]
        pad_num = dict_info["max_obs"] - len(lc_filtered)
        if pad_num >= 0:
            lc_padded = np.pad(
                lc_filtered, (0, pad_num), "constant", constant_values=(0, 0)
            )
        return lc_padded

    elif dict_info["type_windows"] == "linspace_logspace_times":
        if len(flag_detections) == 0:
            pad_num = dict_info["max_obs"] - nepochs
            return np.pad(lc, (0, pad_num), "constant", constant_values=(0, 0))

        idx_first_det = np.argmax(flag_detections)
        # print('idx_first_det: ', idx_first_det)

        mjds_before_det = aux_times[:idx_first_det]
        if len(mjds_before_det) > num_obs_before_det:
            length_forced = mjds_before_det.shape[0]
            linspace_mjd_idxs = np.linspace(
                0, length_forced - 1, num=num_obs_before_det
            ).astype(int)
        else:
            linspace_mjd_idxs = np.arange(idx_first_det)
        # print('linspace_mjd_idxs: ', linspace_mjd_idxs)
        # print('nepochs: ', nepochs)

        mjds_after_det = aux_times[idx_first_det:]
        grid_mjds = np.geomspace(
            0.5, MAX_DAYS, num=dict_info["max_obs"] - linspace_mjd_idxs.shape[0]
        ) + np.min(mjds_after_det)
        logspace_mjd_idxs = find_nearest_unique_times(mjds_after_det, grid_mjds)
        logspace_mjd_idxs += idx_first_det

        # print('logspace_mjd_idxs: ', logspace_mjd_idxs)

        indices = np.concatenate([linspace_mjd_idxs, logspace_mjd_idxs], axis=0)
        lc_filtered = lc[indices]
        pad_num = dict_info["max_obs"] - len(lc_filtered)
        if pad_num >= 0:
            lc_padded = np.pad(
                lc_filtered, (0, pad_num), "constant", constant_values=(0, 0)
            )
        return lc_padded


# Genero la mascara donde hay observaciones de flujos, es decir, donde no hay PAD?
def create_mask(lc):
    return (lc != 0).astype(float)


def normalizing_time(time_fid):
    mask_min = 9999999999 * (time_fid == 0).astype(
        float
    )  # Enmascara el PAD generado en separate_by_filter
    t_min = np.min(time_fid + mask_min)
    return (time_fid - t_min) * (~(time_fid == 0)).astype(
        float
    )  # Considero solo los tiempos donde hay observaciones


# Matrix (Observaciones x filtros) en cada fila por columna secuenciales
# Realiza PAD considerando el Largo Maximo dado de 65 epochs
def separate_by_filter(col_serie, bands, times, detections, dict_info):
    col_array = np.array(col_serie)
    band_array = np.array(bands)

    times_array = np.array(times)
    detections_array = np.array(detections)

    final_array = []
    for i, color in enumerate(dict_info["bands_to_use"]):
        aux = col_array[band_array == color]
        aux_times = times_array[band_array == color]
        aux_detections = detections_array[band_array == color]
        nepochs = len(aux)
        final_array += [pad_list(aux, nepochs, dict_info, aux_times, aux_detections)]

    return np.stack(final_array, 1)


def mask_detection(mask, detection):
    return mask * detection


def mask_photometry(mask, detection):
    return mask * (1 - detection)


def normalizing_time_detection(time, mask_detection):
    t_detections = time * mask_detection
    mask_min = 9999999999 * (mask_detection == 0).astype(float)
    t_min = np.min(t_detections + mask_min)
    return ((t_detections - t_min) * mask_detection).astype(float)


def normalizing_time_photometric(time, mask_non_detection):
    t_phot = time * mask_non_detection
    mask_min = 9999999999 * (mask_non_detection == 0).astype(float)
    t_min = np.min(t_phot + mask_min)
    return ((t_phot - t_min) * mask_non_detection).astype(float)


# Le restamos el primer MJD a todos los tiempos de todas las bandas
def normalizing_time_phot(time, mask_alert):
    mask_min = 999999999999 * (mask_alert == 0).astype(float)
    t_min = np.min(time + mask_min)
    return (time - t_min) * mask_alert


### Auxiliary functions
def intersection_list(important_list, other_list):
    inter = []
    diff = []
    for obj_imp in important_list:
        if obj_imp in other_list:
            inter += [obj_imp]
        else:
            diff += [obj_imp]
    return inter, diff


def check_if_nan_in_list(pd_used, columns):
    nan_cols = []
    for col in columns:
        if np.isnan(pd_used[col].to_numpy().mean()):
            nan_cols += [col]
    return nan_cols


def create_windows(lc, max_obs):
    windows = []
    for inicio in range(0, len(lc), max_obs):
        fin = inicio + max_obs
        window = lc.iloc[inicio:fin]
        windows.append(window)
    return windows


def get_windows(df_chunk_filtered, dict_cols, dict_info):
    list_df_windows = df_chunk_filtered.groupby(dict_cols["oid"]).apply(
        lambda x: create_windows(x, dict_info["max_obs"])
    )

    dict_num_windows = dict()
    lcs_windows = []
    for _, df_windows in list_df_windows.items():
        for j, df_window in enumerate(df_windows):
            dict_num_windows[df_window[dict_cols["oid"]].iloc[0]] = j
            df_window[dict_cols["oid"]] = df_window[dict_cols["oid"]].iloc[
                0
            ] + "_{}".format(j)
            lcs_windows.append(df_window)

    df_chunk_filtered = pd.concat(lcs_windows)
    return df_chunk_filtered, dict_num_windows


def expands_ids(partitions_final, snid, df_partition, dict_snid_windows, dict_cols):
    aux_snid = copy.copy(snid)
    df_aux = df_partition[df_partition[dict_cols["oid"]] == snid]
    num_windows = dict_snid_windows[snid]

    for i in range(num_windows + 1):
        df_aux.loc[
            df_aux[dict_cols["oid"]] == aux_snid, dict_cols["oid"]
        ] = "{}_{}".format(snid, i)
        partitions_final.append(df_aux.copy())
        aux_snid = "{}_{}".format(snid, i)


def processing_lc(path_lcs_file, dict_cols, dict_info, df_objid_label):
    print(
        "Generate different windows for each light curve? {}".format(
            dict_info["type_windows"]
        )
    )

    # Podria realizar multiproceso por chunk, si es que son muchos.
    # dict_snid_windows = dict()
    df_final = []

    path_lcs_chunks = glob.glob("{}/lightcurves*".format(path_lcs_file))
    for path_chunk in tqdm(path_lcs_chunks, "processing lc chunks"):
        
        # Lightcurves
        df_chunk = pd.read_parquet("{}".format(path_chunk))
        df_chunk = df_chunk[
            df_chunk[dict_cols["oid"]].isin(df_objid_label[dict_cols["oid"]].values)
        ]
        df_chunk = (
            df_chunk.groupby(dict_cols["oid"])
            .apply(lambda x: x.sort_values(dict_cols["time"]))
            .reset_index(drop=True)
        )
        df_chunk_filtered = df_chunk[
            list(set(dict_cols.values()) - {dict_cols["class"]})
        ]

        if dict_info["type_windows"] == "windows":
            # df_chunk_filtered, dict_num_windows = get_windows(df_chunk_filtered, dict_cols, dict_info)
            # dict_snid_windows.update(dict_num_windows)
            df_chunk_filtered["window_id"] = (
                df_chunk_filtered.groupby(dict_cols["oid"]).cumcount() // 200
            )
            df_grouped = df_chunk_filtered.groupby([dict_cols["oid"], "window_id"]).agg(
                lambda x: list(x)
            )

        else:
            df_grouped = df_chunk_filtered.groupby([dict_cols["oid"]]).agg(
                lambda x: list(x)
            )

        pd_dataset = pd.DataFrame()
        for col_name in dict_cols.keys():
            if col_name not in ["oid", "band", "class"]:
                pd_dataset[col_name] = df_grouped.apply(
                    lambda x: separate_by_filter(
                        x[dict_cols[col_name]],
                        x[dict_cols["band"]],
                        x[dict_cols["time"]],
                        x[dict_cols["detected"]],
                        dict_info,
                    ),
                    axis=1,
                )

        pd_dataset["time"] = pd_dataset.apply(
            lambda x: normalizing_time(x["time"]), axis=1
        )
        pd_dataset["mask"] = pd_dataset.apply(lambda x: create_mask(x["flux"]), axis=1)

        pd_dataset["mask_detection"] = pd_dataset.apply(
            lambda x: mask_detection(x["mask"], x["detected"]), axis=1
        )
        pd_dataset["time_detection"] = pd_dataset.apply(
            lambda x: normalizing_time_detection(x["time"], x["mask_detection"]), axis=1
        )

        pd_dataset["mask_photometry"] = pd_dataset.apply(
            lambda x: mask_photometry(x["mask"], x["detected"]), axis=1
        )
        pd_dataset["time_photometry"] = pd_dataset.apply(
            lambda x: normalizing_time_photometric(x["time"], x["mask_photometry"]),
            axis=1,
        )

        df_final.append(pd_dataset)

    df_final = pd.concat(df_final)

    return df_final.reset_index()  # , dict_snid_windows
