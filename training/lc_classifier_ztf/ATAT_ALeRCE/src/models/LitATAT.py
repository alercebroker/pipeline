import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchmetrics

import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR
from src.training.schedulers import cosine_decay_ireyes
from src.layers import ATAT


class LitATAT(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.atat = ATAT(**kwargs)
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

        self.use_lightcurves = self.general_["use_lightcurves"]
        self.use_lightcurves_err = self.general_["use_lightcurves_err"]
        self.use_metadata = self.general_["use_metadata"]
        self.use_features = self.general_["use_features"]

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.general_["num_classes"]
        )
        self.train_f1s = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=self.general_["num_classes"], average="macro"
        )
        self.train_rcl = torchmetrics.classification.Recall(
            task="multiclass", num_classes=self.general_["num_classes"], average="macro"
        )

        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.general_["num_classes"]
        )
        self.valid_f1s = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=self.general_["num_classes"], average="macro"
        )
        self.valid_rcl = torchmetrics.classification.Recall(
            task="multiclass", num_classes=self.general_["num_classes"], average="macro"
        )

        self.use_cosine_decay = kwargs["general"]["use_cosine_decay"]
        self.gradient_clip_val = (
            1.0 if kwargs["general"]["use_gradient_clipping"] else 0
        )

    def training_step(self, batch_data, batch_idx):
        input_dict = self.get_input_data(batch_data)

        pred_lc, pred_tab, pred_mix = self.atat(**input_dict)
        pred = (
            pred_mix
            if pred_mix is not None
            else (pred_lc if pred_lc is not None else pred_tab)
        )

        if pred is None:
            raise ValueError("Invalid prediction.")

        """ labels """
        y_true = batch_data["labels"].long()

        self.train_acc(pred, y_true)
        self.train_f1s(pred, y_true)
        self.train_rcl(pred, y_true)

        loss = 0

        loss_dic = {}
        for y, y_type in zip([pred_lc, pred_tab, pred_mix], ["lc", "tab", "mix"]):
            if y is not None:
                partial_loss = F.cross_entropy(y, y_true)
                loss += partial_loss
                loss_dic.update({f"loss_train/{y_type}": partial_loss})

        loss_dic.update({f"loss_train/total": loss})
        self.log_dict(loss_dic)

        self.log("mix/acc_train", self.train_acc, on_step=True, on_epoch=True)
        self.log("mix/f1s_train", self.train_f1s, on_step=True, on_epoch=True)
        self.log("mix/rcl_train", self.train_rcl, on_step=True, on_epoch=True)

        self.log(f"loss_train/total", loss)

        return loss

    def validation_step(self, batch_data, batch_idx):
        input_dict = self.get_input_data(batch_data)

        pred_lc, pred_tab, pred_mix = self.atat(**input_dict)
        pred = (
            pred_mix
            if pred_mix is not None
            else (pred_lc if pred_lc is not None else pred_tab)
        )

        if pred is None:
            raise ValueError("Invalid prediction.")

        """ labels """
        y_true = batch_data["labels"].long()

        self.valid_acc(pred, y_true)
        self.valid_f1s(pred, y_true)
        self.valid_rcl(pred, y_true)

        loss = 0
        loss_dic = {}
        for y, y_type in zip([pred_lc, pred_tab, pred_mix], ["lc", "tab", "mix"]):
            if y is not None:
                partial_loss = F.cross_entropy(y, y_true)
                loss += partial_loss
                loss_dic.update({f"loss_validation/{y_type}": partial_loss})

        loss_dic.update({f"loss_validation/total": loss})
        self.log_dict(loss_dic)

        self.log("mix/acc_valid", self.valid_acc, on_epoch=True)
        self.log("mix/f1s_valid", self.valid_f1s, on_epoch=True)
        self.log("mix/rcl_valid", self.valid_rcl, on_epoch=True)

        return loss_dic

    def test_step(self, batch_data, batch_idx):
        input_dict = self.get_input_data(batch_data)

        pred_lc, pred_tab, pred_mix = self.atat(**input_dict)
        pred = (
            pred_mix
            if pred_mix is not None
            else (pred_lc if pred_lc is not None else pred_tab)
        )

        if pred is None:
            raise ValueError("Invalid prediction.")

        """ labels """
        y_true = batch_data["labels"].long()

        loss = 0
        loss_dic = {}
        for y, y_type in zip([pred_lc, pred_tab, pred_mix], ["lc", "tab", "mix"]):
            partial_loss = F.cross_entropy(y, y_true)
            loss += partial_loss
            loss_dic.update({f"loss_test/{y_type}": partial_loss})

        loss_dic.update({f"loss_test/total": loss})
        self.log_dict(loss_dic)

        return loss_dic

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(params, lr=self.general_["lr"])

        if self.use_cosine_decay:
            scheduler = LambdaLR(
                optimizer,
                lambda epoch: cosine_decay_ireyes(
                    epoch, warm_up_epochs=10, decay_steps=150, alpha=0.05
                ),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        else:
            return optimizer

    def get_input_data(self, batch_data):
        input_dict = {}

        if self.use_lightcurves:
            input_dict.update(
                {
                    "data": batch_data["data"].float(),
                    "time": batch_data["time"].float(),
                    "mask": batch_data["mask"].float(),
                }
            )

        if self.use_lightcurves_err:
            input_dict.update({"data_err": batch_data["data_err"].float()})

        tabular_features = []
        if self.use_metadata:
            tabular_features.append(batch_data["metadata_feat"].float().unsqueeze(2))

        if self.use_features:
            tabular_features.append(batch_data["extracted_feat"].float().unsqueeze(2))

        if tabular_features:
            input_dict["tabular_feat"] = torch.cat(tabular_features, axis=1)

        return input_dict
