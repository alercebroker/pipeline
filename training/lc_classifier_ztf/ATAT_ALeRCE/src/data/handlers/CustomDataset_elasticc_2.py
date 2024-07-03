import numpy as np
import logging

import h5py
import random
import torch

from torch.utils.data import Dataset
from joblib import load


class ATATDataset(Dataset):
    def __init__(
        self,
        data_root="data/final/ZTF_ff/LC_MD_FEAT_v2",
        set_type="train",
        use_lightcurves=True,
        use_metadata=False,
        use_features=False,
        seed=0,
        eval_metric=None,
        force_online_opt=False,
        per_init_time=0.2,
        online_opt_tt=False,
        same_partition=False,
        use_QT=False,
        list_time_to_eval=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        **kwargs,
    ):
        """loading dataset from H5 file"""
        """ dataset is composed for all samples, where self.these_idx dart to sampels for each partition"""

        name = (
            "training"
            if set_type == "train" or set_type == "train_step"
            else "validation"
        )
        partition_used = seed if not same_partition else 0

        h5_ = h5py.File("{}/dataset.h5".format(data_root))

        self.these_idx = (
            h5_.get("test")[:]
            if set_type == "test"
            else h5_.get("%s_%s" % (name, partition_used))[:]
        )

        print(
            f"using set {set_type} total of idx : {len(self.these_idx)}, \
                use_lightcurves {use_lightcurves}, use_metadata {use_metadata}, use_features {use_features}, \
                    use MTA {online_opt_tt}"
        )

        self.data = torch.from_numpy(h5_.get("data")[:][self.these_idx])  # flux
        self.data_var = torch.from_numpy(
            h5_.get("data-var")[:][self.these_idx]
        )  # data-var # flux_err
        self.mask = torch.from_numpy(
            h5_.get("mask_alert")[:][self.these_idx]
        )  # mask_alert # mask
        self.time = torch.from_numpy(
            h5_.get("time_phot")[:][self.these_idx]
        )  # time_phot # time
        # self.time_alert = torch.from_numpy(h5_.get("time_detection")[:][self.these_idx])
        # self.time_phot = torch.from_numpy(h5_.get("time_photometry")[:][self.these_idx])
        self.labels = h5_.get("labels")[:][self.these_idx]

        self.time_alert = torch.from_numpy(
            h5_.get("time_alert")[:][self.these_idx]
        )  # BORRAR

        self.eval_time = eval_metric  # must be a number
        self.max_time = 1500
        self.use_lightcurves = use_lightcurves
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.use_QT = use_QT

        self.set_type = set_type
        self.force_online_opt = force_online_opt
        self.per_init_time = per_init_time
        self.online_opt_tt = online_opt_tt

        self.list_time_to_eval = list_time_to_eval
        print("list_time_to_eval: ", list_time_to_eval)

        logging.info(f"Partition : {partition_used} Set Type : {set_type}")

        if self.use_metadata:
            metadata_feat = h5_.get("metadata_feat")[:][self.these_idx]
            path_QT = "./{}/quantiles/metadata/fold_{}.joblib".format(
                data_root, partition_used
            )
            self.metadata_feat = self.get_tabular_data(
                metadata_feat, path_QT, "metadata"
            )

        if self.use_features:
            self.extracted_feat = dict()
            for time_eval in self.list_time_to_eval:
                path_QT = "./{}/quantiles/features/{}_days/fold_{}.joblib".format(
                    data_root, time_eval, partition_used
                )
                extracted_feat = h5_.get("extracted_feat_{}".format(time_eval))[:][
                    self.these_idx
                ]
                self.extracted_feat.update(
                    {
                        time_eval: self.get_tabular_data(
                            extracted_feat, path_QT, "features"
                        )
                    }
                )

    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construc its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """

        data_dict = {
            "time": self.time[idx],
            "mask": self.mask[idx],
            "labels": self.labels[idx],
            "time_alert": self.time_alert[idx],  # BORRAR
        }

        if self.use_lightcurves:
            data_dict.update({"data": self.data[idx]})

        if self.use_metadata:
            data_dict.update({"metadata_feat": self.metadata_feat[idx]})

        if self.use_features:
            data_dict.update(
                {"extracted_feat": self.extracted_feat[self.list_time_to_eval[-1]][idx]}
            )

        if self.set_type == "train":
            if self.force_online_opt:
                data_dict = self.sc_augmenation(data_dict, idx)
            if self.online_opt_tt:
                data_dict = self.three_time_mask(data_dict, idx)

        return data_dict

    def __len__(self):
        """lenght of the dataaset, is necesary for consistent getitem values"""
        return len(self.labels)

    def get_tabular_data(self, tabular_data, path_QT, type_data):
        logging.info(f"Loading and procesing {type_data}. Using QT: {self.use_QT}")
        if self.use_QT:
            QT = load(path_QT)
            tabular_data = QT.transform(tabular_data)

        return torch.from_numpy(tabular_data)

    def sc_augmenation(self, sample: dict, index: int):
        """sample is a dictionary objg"""
        mask, time_alert = sample["mask"], sample["time"]
        """ random value to asing new light curvee """

        random_value = random.uniform(0, 1)
        max_time = (time_alert * mask).max()
        init_time = self.per_init_time * max_time
        eval_time = init_time + (max_time - init_time) * random_value
        mask_time = (time_alert <= eval_time).float()

        """ if lc features are using in training tablular feeat is updated to especifict time (near to)"""

        if self.use_features:
            """tabular features is updated to acording time span"""
            sample["extracted_feat"] = torch.from_numpy(
                self.add_feat_col_list[
                    "time_%s"
                    % self.list_time_to_eval[
                        (eval_time.numpy() <= self.list_time_to_eval).argmax()
                    ]
                ][index, :]
            )

        """ multiplication of mask, where are both enabled is the final mask """

        sample["mask"] = mask * mask_time
        return sample

    def three_time_mask(self, sample: dict, idx: int):
        """sample is update to a fixed lenght betwent thre values"""
        mask, time = sample["mask"], sample["time_alert"]  # BORRAR (time_alert to time)
        time_eval = np.random.choice(
            [8, 128, 2048]
        )  # 16, 32, 64, 128, 256, 512, 1024, 2048]) # BORRAR
        mask_time = (time <= time_eval).float()

        if self.use_features:
            sample["extracted_feat"] = self.extracted_feat[time_eval][idx]

        sample["mask"] = mask * mask_time

        return sample

    def update_mask(self, sample: dict, timeat: int):
        sample.update(
            {
                "mask": sample["mask"]
                * (sample["time_alert"] - sample["time_alert"][0, :].min() < timeat)
                * (sample["time_photo"] - sample["time_photo"][0, :].min() < timeat)
            }
        )

        return sample
