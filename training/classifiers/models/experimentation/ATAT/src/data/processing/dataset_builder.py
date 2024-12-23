import numpy as np
import h5py

def create_dataset_h5py(all_partitions, all_data, num_folds, dict_info, path_save_dataset):

    with h5py.File("{}/dataset.h5".format(path_save_dataset), "w") as hf:
        hf.create_dataset("SNID", data=np.array(all_data["oid"].to_numpy()))
        hf.create_dataset("time", data=np.stack(all_data["time"].to_list(), 0))
        hf.create_dataset("brightness", data=np.stack(all_data["brightness"].to_list(), 0))
        hf.create_dataset("e_brightness", data=np.stack(all_data["e_brightness"].to_list(), 0))
        hf.create_dataset("mask", data=np.stack(all_data["mask"].to_list(), 0))
        hf.create_dataset(
            "time_detection", data=np.stack(all_data["time_detection"].to_list(), 0)
        )
        hf.create_dataset(
            "mask_detection", data=np.stack(all_data["mask_detection"].to_list(), 0)
        )
        hf.create_dataset(
            "time_photometry", data=np.stack(all_data["time_photometry"].to_list(), 0)
        )
        hf.create_dataset(
            "mask_photometry", data=np.stack(all_data["mask_photometry"].to_list(), 0)
        )

        hf.create_dataset("labels", data=np.stack(all_data["label_int"].to_list(), 0))
        hf.create_dataset("class_name", data=np.array(all_data["class_name"].to_numpy()))

        if dict_info['extract_metadata']:
            hf.create_dataset("metadata_feat", data=np.stack(all_data["metadata_feat"].to_list()))

        if dict_info['extract_features']:
            for time_to_eval in dict_info["list_time_to_eval"]:
                name_dataset = f'extracted_feat_{time_to_eval}'
                hf.create_dataset(name_dataset, data=np.stack(all_data[name_dataset].to_list()))

        for fold in range(num_folds):
            print("- fold: {}".format(fold))
            aux_pd = all_partitions["fold_%s" % fold]

            hf.create_dataset(
                "training_%d" % fold,
                data=aux_pd[
                    aux_pd["partition"] == "training_%d" % fold
                ].index.to_numpy(),
            )
            hf.create_dataset(
                "validation_%d" % fold,
                data=aux_pd[
                    aux_pd["partition"] == "validation_%d" % fold
                ].index.to_numpy(),
            )

        hf.create_dataset(
            "test", data=aux_pd[aux_pd["partition"] == "test"].index.to_numpy()
        )
        hf.close()