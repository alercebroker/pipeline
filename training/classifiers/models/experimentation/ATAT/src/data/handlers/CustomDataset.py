import numpy as np
import logging
import os
import h5py
import random
import torch

import pandas as pd

from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import Dataset
from joblib import load, dump


class ATATDataset(Dataset):
    def __init__(
        self,
        path_results,
        data_root="data/final/ZTF_ff/LC_MD_FEAT_v2",
        set_type="train",
        use_lightcurves=True,
        use_lightcurves_err=False,
        use_metadata=False,
        use_features=False,
        fold=0,
        force_online_opt=False,
        per_init_time=0.2,
        online_opt_tt=False,
        norm_type='None',
        use_QT=False,
        qt_type='qt_global',
        feat_cols=None,
        list_time_to_eval=[16, 32, 64, 128, 256, 512, 1024, None],
        **kwargs,
    ):
        """loading dataset from H5 file"""
        """ dataset is composed for all samples, where self.these_idx dart to samples for each partition"""

        name = (
            "training"
            if set_type == "train" else "validation"
        )

        h5_ = h5py.File("{}/dataset.h5".format(data_root))

        self.these_idx = (
            h5_.get("test")[:]
            if set_type == "test"
            else h5_.get("%s_%s" % (name, fold))[:]
        )

        print(
            f"using set {set_type} total of idx : {len(self.these_idx)}, \
                use_lightcurves {use_lightcurves}, use_metadata {use_metadata}, use_features {use_features}, \
                    use MTA {online_opt_tt}"
        )

        self.path_results = path_results
        self.lcid = h5_.get("SNID")[:][self.these_idx]
        self.data = torch.from_numpy(h5_.get("brightness")[:][self.these_idx]) 
        self.data_err = torch.from_numpy(
            h5_.get("e_brightness")[:][self.these_idx]
        ) 
        self.mask = torch.from_numpy(
            h5_.get("mask")[:][self.these_idx]
        ) 
        self.time = torch.from_numpy(
            h5_.get("time")[:][self.these_idx]
        ) 
        self.time_alert = torch.from_numpy(h5_.get("time_detection")[:][self.these_idx])
        self.time_phot = torch.from_numpy(h5_.get("time_photometry")[:][self.these_idx])
        self.labels = h5_.get("labels")[:][self.these_idx].astype(int)
        self.class_name = h5_.get("class_name")[:][self.these_idx]

        self.use_lightcurves = use_lightcurves
        self.use_lightcurves_err = use_lightcurves_err
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.use_QT = use_QT
        self.qt_type = qt_type

        self.set_type = set_type
        self.force_online_opt = force_online_opt
        self.per_init_time = per_init_time
        self.online_opt_tt = online_opt_tt

        self.list_time_to_eval = list_time_to_eval
        
        if self.use_lightcurves:
            self.apply_normalization(norm_type)

        if self.use_metadata or self.use_features:
            self.metadata_feat = None
            self.extracted_feat = None
            self.mask_metadata = None
            self.mask_feat = None
            self.get_tabular_data(h5_, set_type, feat_cols, fold)

        logging.info(f"Processing partition: {fold} - with set type: {set_type}...")

    def __len__(self):
        """length of the dataset, is necessary for consistent getitem values"""
        return len(self.labels)

    def __getitem__(self, idx):
        """idx is used for pytorch to select samples to construct its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """

        data_dict = {
            'idx': idx,
            "id": self.lcid[idx],
            "time": self.time[idx],
            "mask": self.mask[idx],
            "labels": self.labels[idx],
        }

        if self.use_lightcurves:
            data_dict.update({"data": self.data[idx]})

        if self.use_lightcurves_err:
            data_dict.update({"data_err": self.data_err[idx]})

        if self.use_metadata:
            data_dict.update({
                "metadata_feat": self.metadata_feat[idx],
                "mask_metadata": self.mask_metadata[idx],
                })

        if self.use_features:
            data_dict.update({
                "extracted_feat": self.extracted_feat[self.list_time_to_eval[-1]][idx],
                "mask_feat": self.mask_feat[self.list_time_to_eval[-1]][idx],
                })

        if self.set_type == "train":
            if self.force_online_opt:
                data_dict = self.sc_augmenation(data_dict, idx)
            if self.online_opt_tt:
                data_dict = self.three_time_mask(data_dict, idx)

        return data_dict

    def sc_augmenation(self, sample: dict, index: int):
        """sample is a dictionary obj"""
        mask, time_alert = sample["mask"], sample["time"]
        """ random value to asing new light curve """

        random_value = random.uniform(0, 1)
        max_time = (time_alert * mask).max()
        init_time = self.per_init_time * max_time
        eval_time = init_time + (max_time - init_time) * random_value
        mask_time = (time_alert <= eval_time).float()

        """ if lc features are using in training tabular feeat is updated to specific time (near to)"""

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
        """Aplica una máscara a la muestra en función de un límite de tiempo aleatorio."""
        mask, time = sample["mask"], sample["time"]
        time_eval = np.random.choice([16, 128, None]) 
        
        if time_eval is not None:
            mask_time = (time <= time_eval).float()
        else:
            mask_time = torch.ones_like(time)
        
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

    def apply_normalization(self, norm_type):
        if norm_type == 'None':
            pass
        elif norm_type == 'zero_mean':
            self.data = self.data - self.data.mean(dim=1, keepdim=True)
            self.data_err = self.data_err - self.data_err.mean(dim=1, keepdim=True)
        elif norm_type == 'arcsinh':
            scaling_factor = 1.0
            self.data_err = self.data_err / torch.sqrt((self.data / scaling_factor) ** 2 + 1)
            self.data = torch.arcsinh(self.data / scaling_factor)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}. "
                             f"Valid options are: 'None', 'zero_mean', 'arcsinh'.")

    def calculate_and_save_QT(self, tabular_data, path_QT):
        qt = QuantileTransformer(
            n_quantiles=1000, random_state=0, output_distribution="uniform"
        )
        qt.fit(tabular_data)
        dump(qt, f"{path_QT}")

    def apply_QT(self, tabular_data, path_QT, tab_type):
        logging.info(f"Loading and procesing {tab_type}. Using QT: {self.use_QT}")
        QT = load(path_QT)
        transformed_values = QT.transform(tabular_data)
        tabular_data[:] = transformed_values
        tabular_data += 0.1
        return tabular_data

    def manage_bias_periodic_others(self, tabular_data, feat_cols):   
        df_objid_label = pd.read_parquet('../../../data_acquisition/ztf_forced_photometry/raw/objects.parquet').reset_index()
        coordinate_cols = [f"Coordinate_{x}" for x in "xyz"]
        tabular_data = {
            "lcid": self.lcid,
            "oid": [oid.split(b'_')[0].decode('utf-8') for oid in self.lcid], 
            **{col: tabular_data[:, idx] for idx, col in enumerate(feat_cols)},
        }
        tabular_data = pd.DataFrame(tabular_data)

        lcids_po_tp = self.lcid[self.class_name == b"Periodic-Other"]
        training_po_oids = list(set([oid.split(b'_')[0].decode('utf-8') for oid in lcids_po_tp]))
        po_tp_coords = df_objid_label[df_objid_label["oid"].isin(training_po_oids)]
        southern = po_tp_coords["dec"] < -20

        # Estimates how many southern objects should not be replaced based on a scaling factor 
        # that considers geographical bounds (declination ranges) and the proportions of northern objects.
        n_not_to_be_replaced = ((~southern).astype(float).sum()) / (54 - (-20)) * (28 - 20)
        n_not_to_be_replaced = int(np.ceil(n_not_to_be_replaced))

        southern_po_oids = po_tp_coords[southern]["oid"].values
        northern_po_oids = po_tp_coords[~southern]["oid"].values

        np.random.seed(0)
        southern_not_to_be_replaced = np.random.choice(
            southern_po_oids, size=n_not_to_be_replaced, replace=False
        )
        not_to_be_replaced = np.concatenate([northern_po_oids, southern_not_to_be_replaced])
        to_be_replaced = list(set(training_po_oids) - set(not_to_be_replaced))

        # We replace the same coordinates for all the windows of the same object.
        df_reduced = (
            tabular_data 
            .groupby("oid") 
            [coordinate_cols] 
            .mean() 
            .reset_index()
        )
        tbr_feature_mask = df_reduced["oid"].isin(to_be_replaced)
        n_replacement_needed = tbr_feature_mask.astype(int).sum()
        replacement_coords = (
            df_reduced[df_reduced["oid"].isin(not_to_be_replaced)][
                coordinate_cols
            ]
            .sample(n_replacement_needed, replace=True, random_state=0)
            .values
        )
        df_reduced = df_reduced.set_index("oid")
        df_reduced.loc[
            to_be_replaced, coordinate_cols
        ] = replacement_coords

        tabular_data.set_index("oid", inplace=True)
        tabular_data = tabular_data.drop(['lcid'] + coordinate_cols, axis=1)
        tabular_data = pd.merge(tabular_data, df_reduced, on='oid')[feat_cols].values

        return tabular_data

    def get_metadata(self, h5_, set_type, fold, tab_type):
        metadata_feat = h5_.get("metadata_feat")[:][self.these_idx]
        metadata_feat = np.where(np.isinf(metadata_feat), np.nan, metadata_feat) ############ <---- np.nan

        if self.use_QT:
            path_QT = self._get_qt_path(tab_type, fold)
            if set_type == 'train':
                self.calculate_and_save_QT(tabular_data=metadata_feat, path_QT=path_QT)

            metadata_feat = self.apply_QT(metadata_feat, path_QT, tab_type)

        self.mask_metadata = torch.from_numpy(~np.isnan(metadata_feat))
        self.metadata_feat = torch.from_numpy(np.nan_to_num(metadata_feat, nan=0.0)) 

    def get_features(self, h5_, set_type, feat_cols, fold, tab_type):
        time_eval = self.list_time_to_eval[-1]
        extracted_feat = h5_.get("extracted_feat_{}".format(time_eval))[:][
            self.these_idx
        ]
        extracted_feat = np.where(np.isinf(extracted_feat), np.nan, extracted_feat)  ############ <---- np.nan
        
        if set_type == "train":
            extracted_feat = self.manage_bias_periodic_others(extracted_feat, feat_cols)
        
        if self.use_QT:
            path_QT = self._get_qt_path(tab_type, fold, time_eval)
            if set_type == 'train':
                self.calculate_and_save_QT(tabular_data=extracted_feat, path_QT=path_QT)

        self.extracted_feat = dict()
        self.mask_feat = dict()
        for time_eval in self.list_time_to_eval:
            extracted_feat = h5_.get("extracted_feat_{}".format(time_eval))[:][
                self.these_idx
            ]
            extracted_feat = np.where(np.isinf(extracted_feat), np.nan, extracted_feat) ############ <---- np.nan

            if set_type == "train":
                extracted_feat = self.manage_bias_periodic_others(extracted_feat, feat_cols)

            if self.use_QT:
                if self.qt_type == 'qt_global':
                    extracted_feat = self.apply_QT(extracted_feat, path_QT, f"{tab_type}_{time_eval}")
                elif self.qt_type == 'qt_per_day':
                    path_QT = self._get_qt_path(tab_type, fold, time_eval)
                    if set_type == 'train':
                        self.calculate_and_save_QT(tabular_data=extracted_feat, path_QT=path_QT)
                    extracted_feat = self.apply_QT(extracted_feat, path_QT, f"{tab_type}_{time_eval}")

            mask_feat = torch.from_numpy(~np.isnan(extracted_feat))
            self.mask_feat.update({time_eval: mask_feat})

            extracted_feat = torch.from_numpy(np.nan_to_num(extracted_feat, nan=0.0))
            self.extracted_feat.update({time_eval: extracted_feat})

    def get_tabular_data(self, h5_, set_type, feat_cols, fold):
        if self.use_metadata:
            self.get_metadata(h5_, set_type, fold, tab_type='metadata')

        if self.use_features:
            self.get_features(h5_, set_type, feat_cols, fold, tab_type='features')

    def _get_qt_path(self, tab_type, fold, time_eval=None):
        """
        Constructs the path for the Quantile Transformer based on parameters.
        """
        if tab_type == 'metadata':
            qt_dir = os.path.join(self.path_results, "quantiles", tab_type)
        elif tab_type == 'features':
            qt_dir = os.path.join(self.path_results, "quantiles", tab_type, f"{time_eval}_days")
        os.makedirs(qt_dir, exist_ok=True)
        return os.path.join(qt_dir, f"fold_{fold}.joblib")

    def obtain_valid_mask(self, sample, time_eval, idx):
        if self.use_lightcurves:
            mask, time = sample["mask"], sample["time"]
            if time_eval is not None:
                mask_time = (time <= time_eval).float()
                sample["mask"] = mask * mask_time

        if self.use_features:
            sample['extracted_feat'] = self.extracted_feat[time_eval][idx]
            sample['mask_feat'] = self.mask_feat[time_eval][idx]

        return sample