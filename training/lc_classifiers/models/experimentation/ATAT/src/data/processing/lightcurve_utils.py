import numpy as np
import pandas as pd

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

    elif dict_info["type_windows"] == "max":
        max_obs = dict_info["max_obs"]  # This is now calculated per object
        pad_num = max_obs - nepochs
        if pad_num > 0:
            return np.pad(lc, (0, pad_num), "constant", constant_values=(0, 0))
        return np.array(lc)  # No padding needed if already at max_obs

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
    return lc != 0 #.astype(float)


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
    return mask & (detection == 1)


def mask_photometry(mask, detection):
    return mask & (detection == 0)


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


def adapting_format_to_lc_windows(df_dataset, df_partitions, num_folds):
    print(
        "We are expanding the number of windows per light curve in test partition..."
    )

    name_cols = ['oid', 'window_id', 'class_name', 'partition', 'label_int']
    partitions_final = []

    test_partition = df_partitions[df_partitions["partition"] == "test"]
    df_partition_expand = pd.merge(
        test_partition, df_dataset, on="oid", how="outer"
    ).dropna()[name_cols]
    df_partition_expand["window_id"] = df_partition_expand["window_id"].astype(int)
    partitions_final.append(df_partition_expand)

    for fold in range(num_folds):
        print(
            "We are expanding the number of windows per light curve in fold {}...".format(
                fold
            )
        )
        this_partitions = df_partitions[
            (df_partitions["partition"] == "training_%d" % fold)
            | (df_partitions["partition"] == "validation_%d" % fold)
        ]

        df_partition_expand = pd.merge(
            this_partitions, df_dataset, on="oid", how="outer"
        ).dropna()[name_cols]
        df_partition_expand["window_id"] = df_partition_expand["window_id"].astype(
            int
        )
    
        partitions_final.append(df_partition_expand)

    df_partitions = pd.concat(partitions_final)
    df_partitions["oid"] = (
        df_partitions["oid"].astype(str)
        + "_"
        + df_partitions["window_id"].astype(str)
    )
    df_partitions = df_partitions.drop(columns=["window_id"])

    df_dataset["oid"] = (
        df_dataset["oid"].astype(str) + "_" + df_dataset["window_id"].astype(str)
    )
    df_dataset = df_dataset.drop(columns=["window_id"])
    df_dataset = df_dataset[df_dataset.oid.isin(df_partitions.oid.unique())]

    return df_dataset, df_partitions