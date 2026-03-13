import h5py
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
import logging
import sys


class ElasticcDataset(Dataset):
    def __init__(
        self,
        data_root="",
        dataset="",
        set_type="train",
        transform=None,
        target_transform=None,
        in_memory=False,
        use_metadata=False,
        use_features=False,
        seed=0,
        eval_metric=None,
        list_eval_metrics=None,
        force_online_opt=False,
        per_init_time=0.2,
        use_mask_alert=False,
        use_small_subset=False,
        use_time_alert=False,
        use_time_phot=False,
        online_opt_tt=False,
        predict_obj="lc",
        F_max=[],
        label_per=0.0,
        same_partition=False,
        not_quantile_transformer=False,
        supernova=False,
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
        name_target = "" if not dataset == "ELASTICC_STREAM" else "test"

        h5_ = h5py.File(
            "%s/%s"
            % (
                data_root,
                "dataset.h5"  # "elasticc_dataset_update_sn.h5"
                if supernova
                else "dataset.h5",  # "elasticc_dataset_update.h5",
            )
        )

        self.these_idx = (
            h5_.get("test")[:]
            if set_type == "test"
            else h5_.get("%s_%s" % (name, partition_used))[:]
        )

        print(
            f"using set {set_type } total of idx : {len(self.these_idx)} , use_metadata {use_metadata}, use_features {use_features}, use MTA {online_opt_tt} sn_experiment {supernova}"
        )

        self.data = h5_.get("data")
        self.data_var = h5_.get("data-var")
        self.mask = h5_.get("mask_alert")  # always use mask alert
        self.mask_detection = h5_.get("mask_detection")
        self.time = h5_.get("time_phot")  # always use time phot
        self.time_alert = h5_.get("time_alert")
        self.time_phot = h5_.get("time_phot")
        self.target = h5_.get("labels")
        self.feat_col = h5_.get("norm_feat_col")

        self.labels = torch.from_numpy(self.target[:][self.these_idx])
        self.eval_time = eval_metric  # must be a number
        self.max_time = 1500
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.set_type = set_type
        self.force_online_opt = force_online_opt
        self.online_opt_tt = online_opt_tt
        self.per_init_time = per_init_time
        self.code_eval_time = np.array(
            ["8", "16", "32", "64", "128", "256", "512", "1024", "2048"]
            # ["0008", "0016", "0032", "0064", "0128", "0256", "0512", "1024", "2048"]
        )
        self.time_eval_time = self.code_eval_time.astype(int)

        # loading quatiles trasnformes
        if self.use_metadata:
            logging.info(
                f"Loading and procesing header features for QTF,   Partition : {partition_used} Set Type : {set_type}"
            )
            self.metadata_qt = load(
                "%s/QT-New%s/md_fold_%s.joblib"
                % (data_root, name_target, partition_used)
            )
            self.feat_col = self.metadata_qt.transform(self.feat_col[:])
            logging.info(f"Normalized header features")

        if self.use_features:
            # add feat _col refers to feeatures
            logging.info(
                f"Loading and procesing extra features for QTF,   Partition : {partition_used} Set Type : {set_type}"
            )
            self.add_feat_col = h5_.get(
                "norm_add_feat_col_2048"  # "feats_2048"
            )  # if self.eval_time is None else 'norm_add_feat_col_%s' % self.eval_time)
            self.features_qt = load(
                "%s/QT-New%s/qt-feat-%s.joblib"
                % (data_root, name_target, partition_used)
            )
            self.add_feat_col = self.features_qt.transform(self.add_feat_col[:])
            logging.info(f"Normalized extra features")
            # if eval time is a list and not None
            self.add_feat_col_list = {}
            for c_, t_ in zip(self.code_eval_time, self.time_eval_time):
                logging.info(f"Procesing feat cols for time {t_}")
                self.add_feat_col_list["time_%s" % t_] = self.features_qt.transform(
                    h5_.get("norm_add_feat_col_%s" % c_)[
                        :
                    ]  # h5_.get("feats_%s" % c_)[:]
                )

    """ data augmentation methods """

    def sc_augmenation(self, sample: dict, index: int):
        """sample is a dictionary objg"""
        mask, time_alert = sample["mask"], sample["time_alert"]
        """ random value to asing new light curvee """

        random_value = random.uniform(0, 1)
        max_time = (time_alert * mask).max()
        init_time = self.per_init_time * max_time
        eval_time = init_time + (max_time - init_time) * random_value
        mask_time = (time_alert <= eval_time).float()

        """ if lc features are using in training tablular feeat is updated to especifict time (near to)"""

        if self.use_features:
            """tabular features is updated to acording time span"""
            sample["add_tabular_feat"] = torch.from_numpy(
                self.add_feat_col_list[
                    "time_%s"
                    % self.time_eval_time[
                        (eval_time.numpy() <= self.time_eval_time).argmax()
                    ]
                ][index, :]
            )

        """ multiplication of mask, where are both enabled is the final mask """

        sample["mask"] = mask * mask_time
        return sample

    """ data augmentation method """

    def three_time_mask(self, sample: dict, index: int):
        """sample is update to a fixed lenght betwent thre values"""

        mask, time_alert = sample["mask"], sample["time_alert"]
        eval_time = np.random.choice([8, 128, 2048])
        mask_time = (time_alert <= eval_time).float()

        if self.use_features:
            sample["add_tabular_feat"] = torch.from_numpy(
                self.add_feat_col_list[
                    "time_%s"
                    % self.time_eval_time[(eval_time <= self.time_eval_time).argmax()]
                ][index, :]
            )

        sample["mask"] = mask * mask_time

        return sample

    """ obtain validad mask for evaluation prupouses"""

    def obtain_valid_mask(self, sample, mask, time_alert, index):
        mask_time = (time_alert <= self.eval_time).float()
        sample["mask"] = mask * mask_time
        if self.use_features:
            sample["add_tabular_feat"] = torch.from_numpy(
                self.add_feat_col_list[
                    "time_%s"
                    % self.list_eval_time[
                        (self.eval_time <= self.list_eval_time).argmax()
                    ]
                ][index, :]
            )
        return sample

    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construc its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """

        idx_ = self.these_idx[idx]

        """ model input is based on a dic of tensors """
        """ data : LC FLUXES """
        """ data_var : LC ERRORS"""
        """ time : LC TIME """
        """ labels: LC LABEL"""
        """ mask : LC MASK 4 ATTENTIONS MECHANIMS """
        """ mask_detection : None"""

        data_dict = {
            "data": torch.from_numpy(self.data[idx_, :, :]),
            "data_var": torch.from_numpy(self.data_var[idx_, :, :]),
            "time": torch.from_numpy(self.time[idx_, :, :]),
            "labels": torch.from_numpy(np.array(self.target[idx_])),
            "mask": torch.from_numpy(self.mask[idx_, :, :]),
            "mask_detection": torch.from_numpy(self.mask_detection[idx_, :, :]),
            "time_alert": torch.from_numpy(self.time_alert[idx_, :, :]),
            "idx": idx_,
        }

        """ using or not metadata, in case of using metadata data_dict is updated with new keys """

        # forward over quatiles transform
        if self.use_metadata:
            try:
                """if using metadata , this mustb be normalized ins sence of quantile"""
                # logging.info(f'{self.feat_col[idx_, : ].shape}')
                # logging.info(f'{(self.feat_col[idx_, : ].reshape(1, 64)).shape}')
                data_qt_md = self.feat_col[idx_, :]
                data_dict.update({"tabular_feat": torch.from_numpy(data_qt_md)})
            except Exception as error:
                logging.info("Error in loading quantiles or metadata from dataset")
                logging.info(f"Details: {error}")
                sys.exit(1)

        """ two choices, evaluating over time or training"""

        # if self.eval_time and self.set_type != "train":
        #     """only for evaluation prupouses"""
        #     data_dict = self.obtain_valid_mask(
        #         data_dict, data_dict["mask"], data_dict["time_alert"], idx_
        #     )

        #     return data_dict, torch.LongTensor(
        #         torch.from_numpy(np.array(self.target[idx_]))
        #     )

        # else:
        #     """only for train prouposes"""
        if self.use_features:
            try:
                """if using features , this mustb be normalized ins sence of quantile"""
                data_qt_fe = self.add_feat_col[idx_, :]
                data_dict.update({"add_tabular_feat": torch.from_numpy(data_qt_fe)})

            except Exception as error:
                logging.info("Error in loading quantiles or features from dataset")
                logging.info(f"Details: {error}")

        """ for training propoused two augmentation modes can be enabled """
        if self.set_type == "train":
            if self.force_online_opt:
                data_dict = self.sc_augmenation(data_dict, idx_)
            if self.online_opt_tt:
                data_dict = self.three_time_mask(data_dict, idx_)

        return (
            data_dict,
            torch.LongTensor(torch.from_numpy(np.array(self.target[idx_]))),
        )

    def __len__(self):
        """lenght of the dataaset, is necesary for consistent getitem values"""
        return len(self.labels)
