import numpy as np
import logging

def prepare_model_input(stamps, max_h, max_w):
    padded_stamps = []
    padding_masks = []

    for row in stamps:  # row = [stamp_1, stamp_2, stamp_3] (cada uno 2D)
        padded_row = []
        mask_row = []
        for stamp in row:
            padded, mask = pad_stamp_and_mask(stamp, max_h, max_w)
            padded_row.append(padded)  # (H, W)
            mask_row.append(mask)      # (H, W)
        padded_stamps.append(np.stack(padded_row, axis=-1))      # (H, W, 3)
        padding_masks.append(np.stack(mask_row, axis=-1))        # (H, W, 3)

    return np.stack(padded_stamps), np.stack(padding_masks)  


def normalize_stamps(stamps_ndarray, padding_mask):
    # Reshape para normalizar por canal
    new_shape = (
        stamps_ndarray.shape[0],
        stamps_ndarray.shape[1] * stamps_ndarray.shape[2],
        stamps_ndarray.shape[3])
    
    # Sustituir valores problemáticos por NaN
    abs_ndarray = np.abs(stamps_ndarray.reshape(new_shape))
    abs_ndarray[np.isinf(abs_ndarray)] = np.nan
    abs_ndarray[abs_ndarray > 1e10] = np.nan

    # Crear máscara de valores inválidos
    is_infinite = (~np.isfinite(stamps_ndarray)).astype(float)

    # Crear máscara final: 1 si el valor es padding O inválido
    final_mask = 1-padding_mask#np.clip(is_infinite + (1.0 - padding_mask), 0, 1)

    # Cálculo de min y percentil 99 por imagen y canal
    maxval = np.nanpercentile(abs_ndarray, 99, axis=1)
    minval = np.nanmin(abs_ndarray, axis=1)

    
    # Normalización per-channel, per-image
    stamps_ndarray = (stamps_ndarray - minval[:, np.newaxis, np.newaxis, :]) / (
        maxval[:, np.newaxis, np.newaxis, :] - minval[:, np.newaxis, np.newaxis, :] + 1e-8)

    # Clipping en caso de valores extremos
    stamps_ndarray = np.clip(stamps_ndarray, a_min=-2.0, a_max=2.0)

    # Reemplazo de valores NaN/infinito por 0
    stamps_ndarray = np.nan_to_num(stamps_ndarray, posinf=0.0, neginf=0.0)

    # Agregar canal con la máscara de invalidez
    stamps_ndarray = np.concatenate([stamps_ndarray, final_mask], axis=3)
    return stamps_ndarray

"""def normalize_stamps(stamps_ndarray, padding_mask):
    # Reshape para normalizar por canal
    new_shape = (
        stamps_ndarray.shape[0],
        stamps_ndarray.shape[1] * stamps_ndarray.shape[2],
        stamps_ndarray.shape[3])
    
    # Sustituir valores problemáticos por NaN
    abs_ndarray = np.abs(stamps_ndarray.reshape(new_shape))
    abs_ndarray[np.isinf(abs_ndarray)] = np.nan
    abs_ndarray[abs_ndarray > 1e10] = np.nan

    # Crear máscara de valores inválidos
    is_infinite = (~np.isfinite(stamps_ndarray)).astype(float)

    # Crear máscara final: 1 si el valor es padding O inválido
    final_mask = np.clip(is_infinite + (1.0 - padding_mask), 0, 1)

    # Cálculo de min y percentil 99 GLOBAL por canal
    flat_abs = abs_ndarray.reshape(-1, abs_ndarray.shape[-1])  # (B*H*W, C)
    maxval = np.nanpercentile(flat_abs, 99, axis=0)  # shape: (C,)
    minval = np.nanmin(flat_abs, axis=0)             # shape: (C,)

    print(stamps_ndarray.shape)
    
    # Normalización global per-channel (todos los ejemplos usan el mismo min/max)
    stamps_ndarray = (stamps_ndarray - minval[None, None, None, :]) / (
        maxval[None, None, None, :] - minval[None, None, None, :] + 1e-8)

    # Clipping en caso de valores extremos
    stamps_ndarray = np.clip(stamps_ndarray, a_min=-2.0, a_max=2.0)

    # Reemplazo de valores NaN/infinito por 0
    stamps_ndarray = np.nan_to_num(stamps_ndarray, posinf=0.0, neginf=0.0)

    # Agregar canal con la máscara de invalidez
    stamps_ndarray = np.concatenate([stamps_ndarray, final_mask], axis=3)
    return stamps_ndarray"""




def normalize_batches(stamps_ndarray, padding_mask, batch_size):
    n_batches = int(np.ceil(len(stamps_ndarray) / batch_size))
    all_batches = []

    for batch_idx in range(n_batches):
        batch = stamps_ndarray[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_mask = padding_mask[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch = normalize_stamps(batch, batch_mask)
        all_batches.append(batch)

    return np.concatenate(all_batches, axis=0)


def crop_stamps_ndarray(stamps, crop_size):
    """ Crops the stamps to a smaller size of crop_size x crop_size pixels.
    stamps must be b01c. The original center might be slightly shifted if crop_size is even."""
    assert stamps.shape[1] == stamps.shape[2]

    delta = int(np.ceil((stamps.shape[1] - crop_size) / 2))
    cropped_stamps = stamps[:, delta:(delta+crop_size), delta:(delta+crop_size), :]

    assert cropped_stamps.shape[1:3] == (crop_size, crop_size)
    return cropped_stamps


def get_max_hw(stamps):
    max_h, max_w = 0, 0
    for row in stamps:  # row: [science, reference, difference]
        for stamp in row:
            h, w = stamp.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
    return max_h, max_w


def pad_stamp_and_mask(stamp, target_h, target_w):
    h, w = stamp.shape
    target_size = 31

    if h != w:
        logging.info(f"Stamp was not square ({h} != {w}), padding to biggest dimension")
        size = max(h, w)
        pad_h = size - h
        pad_w = size - w

        # pad_width = ((arriba, abajo), (izquierda, derecha))
        stamp = np.pad(
            stamp,
            pad_width=((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )

    h, w = stamp.shape

    original_stamp_size = h
    if original_stamp_size > target_size:  # crop
        start_index = (original_stamp_size - target_size) // 2
        end_index = start_index + target_size
        padded_stamp = stamp[start_index:end_index, start_index:end_index]
        mask = np.ones((target_size, target_size), dtype=np.float32)
        return padded_stamp, mask
    else:
        pad_before = (target_size - original_stamp_size) // 2
        pad_after = target_size - original_stamp_size - pad_before

        padded_stamp = np.pad(
            stamp,
            pad_width=((pad_before, pad_after), (pad_before, pad_after)),
            mode="constant",
            constant_values=0,
        )

        # mask has zeros in the "fake" padding area
        # and ones in the original stamp area
        mask = np.pad(
            np.ones((h, w), dtype=np.float32),
            pad_width=((pad_before, pad_after), (pad_before, pad_after)),
            mode="constant",
            constant_values=0,
        )

        return padded_stamp, mask