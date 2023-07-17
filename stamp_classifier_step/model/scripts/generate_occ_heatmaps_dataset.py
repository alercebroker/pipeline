# %%
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from modules.occlusion_generator import OcclusionGenerator
import numpy as np
from modules.data_loaders.frame_to_input import FrameToInput
from parameters import param_keys
from tqdm import tqdm
import pickle as pkl


# this function takes a list of heatmaps and merges them into one single heatmap
def post_process(heatmap):
    # postprocessing
    total = heatmap[0]
    for val in heatmap[1:]:
        total = total + val
    total = total / np.max(total)
    return total


#### This is the meat of the program. It generates the heatmap for the given image location ####


def gen_heatmap(sample, model: DeepHiTSNanNormCropStampModel, box, step, verbose=False):
    sample_size = sample.shape[0]
    pad_size = box.shape[0] - 1

    preds = model.predict_proba(sample[np.newaxis, ...])  # [np.newaxis, ...])
    pred_class_index = np.argmax(preds[0])
    # print(pred_class_index)
    # print(preds)
    pred_class_proba = preds[0][pred_class_index]

    # load correct label text
    ztf_labels = ["AGN", "SN", "VS", "asteroid", "bogus"]
    pred_class_name = ztf_labels[pred_class_index]

    # generate occluded images and location of mask

    occluder = OcclusionGenerator(sample, box, step)
    occluded_sample, occlusion_locations = occluder.gen_minibatch()
    occluded_predictions = model.predict_proba(occluded_sample)

    if verbose:
        print("processng %i images" % occluded_predictions.shape[0])
    heatmap = []
    # unpack prediction values
    for occluded_sample_idx in range(occluded_sample.shape[0]):
        score = occluded_predictions[occluded_sample_idx][pred_class_index]
        r, c = occlusion_locations[occluded_sample_idx]
        scoremap = np.zeros(
            (sample.shape[0] + pad_size * 2, sample.shape[1] + pad_size * 2)
        )
        # print(r,c,occluder.box_size)
        scoremap[r : r + occluder.box_size, c : c + occluder.box_size] = score
        # print(scoremap.shape)
        scoremap = scoremap[
            pad_size : pad_size + sample_size, pad_size : pad_size + sample_size
        ]
        heatmap.append(scoremap)

    return heatmap, pred_class_name, pred_class_proba


def get_box(box_size, sample):
    box = np.ones([box_size, box_size, sample.shape[-1]]) * np.median(
        sample, axis=[0, 1]
    )
    return box


if __name__ == "__main__":
    box_size = 3
    box_step = 1
    save_name = "preprocessed_data_with_heatmaps21x21"
    save_path = "../../pickles/%s.pkl" % save_name

    # data_path = os.path.join("../../pickles", 'training_set_with_bogus.pkl')
    data_path = "../../pickles/converted_data.pkl"
    checkpoint_path = os.path.join(
        PROJECT_PATH, "results/best_model_so_far/checkpoints", "model"
    )

    n_classes = 5
    params = {
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.KERNEL_SIZE: 3,
        param_keys.BATCHNORM_FC: None,
        param_keys.BATCHNORM_CONV: None,
        param_keys.DROP_RATE: 0.5,
        param_keys.BATCH_SIZE: 50,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
    }

    model = DeepHiTSNanNormCropStampModel(params)
    model.load_model(checkpoint_path)

    data_loader = FrameToInput(model.params)
    data_loader.dataset_preprocessor = model.dataset_preprocessor
    # data_dict = data_loader.get_dict()
    dataset = data_loader.get_preprocessed_dataset_unsplitted()
    # count labels
    unique, counts = np.unique(dataset.data_label, return_counts=True)
    print(dict(zip(unique, counts)))

    # generate heatmaps for whole dataset
    heatmaps_list = []
    for sample_i in tqdm(range(dataset.data_array.shape[0])):
        sample = dataset.data_array[sample_i]
        box = get_box(box_size, sample)
        sample_incomplete_heatmaps_list, pred_name, pred_proba = gen_heatmap(
            sample, model, box, box_step
        )
        processed_hm = post_process(sample_incomplete_heatmaps_list)
        heatmaps_list.append(processed_hm)
        # save checkpoint of heatmaps
        if sample_i % 1000 == 0:
            heatmaps_aux = np.array(heatmaps_list)
            data_with_heatmaps_aux = {
                "images": dataset.data_array,
                "labels": dataset.data_label,
                "heatmaps": heatmaps_aux,
            }
            pkl.dump(data_with_heatmaps_aux, open(save_path, "wb"), protocol=2)
    heatmaps = np.array(heatmaps_list)

    # this stores preprocessed data, as heatmaps are model specific, so avoid
    # repreprocessing them in autoencoder
    data_with_heatmaps = {
        "images": dataset.data_array,
        "labels": dataset.data_label,
        "heatmaps": heatmaps,
    }
    pkl.dump(data_with_heatmaps, open(save_path, "wb"), protocol=2)
