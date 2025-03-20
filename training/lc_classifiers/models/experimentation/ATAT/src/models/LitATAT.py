import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchmetrics

import lightning as L

from torch.optim.lr_scheduler import LambdaLR
from src.training.schedulers import cosine_decay_ireyes
from src.layers import ATAT


class LitATAT(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.atat = ATAT(**kwargs)
        self.general_ = kwargs["general"]
        self.lightcv_ = kwargs["lc"]
        self.feature_ = kwargs["ft"]

        self.use_lightcurves = self.general_["use_lightcurves"]
        self.use_lightcurves_err = self.general_["use_lightcurves_err"]
        self.use_metadata = self.general_["use_metadata"]
        self.use_features = self.general_["use_features"]

        self.list_time_to_eval = self.general_["list_time_to_eval"]

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

        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.general_["num_classes"]
        )
        self.test_f1s = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=self.general_["num_classes"], average="macro"
        )
        self.test_rcl = torchmetrics.classification.Recall(
            task="multiclass", num_classes=self.general_["num_classes"], average="macro"
        )

        #for time_to_eval in self.list_time_to_eval:
        #    setattr(self, f"test_acc_{time_to_eval}", torchmetrics.classification.Accuracy(
        #        task="multiclass", num_classes=self.general_["num_classes"]
        #    ))
        #    setattr(self, f"test_f1s_{time_to_eval}", torchmetrics.classification.F1Score(
        #        task="multiclass", num_classes=self.general_["num_classes"], average="macro"
        #    ))
        #    setattr(self, f"test_rcl_{time_to_eval}", torchmetrics.classification.Recall(
        #        task="multiclass", num_classes=self.general_["num_classes"], average="macro"
        #    ))

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
    

    def test_step(self, batch_data, batch_idx, dataloader_idx=None):
        time_to_eval = self.list_time_to_eval[dataloader_idx]

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

        self.test_acc(pred, y_true)
        self.test_f1s(pred, y_true)
        self.test_rcl(pred, y_true)

        loss = 0
        loss_dic = {}
        for y, y_type in zip([pred_lc, pred_tab, pred_mix], ["lc", "tab", "mix"]):
            if y is not None:
                partial_loss = F.cross_entropy(y, y_true)
                loss += partial_loss
                loss_dic.update({f"loss_test/{time_to_eval}_days/{y_type}": partial_loss})

        loss_dic.update({f"loss_test/{time_to_eval}_days/total": loss})
        self.log_dict(loss_dic, add_dataloader_idx=False)

        self.log(f"mix/acc_test/{time_to_eval}_days", self.test_acc, on_epoch=True, add_dataloader_idx=False)
        self.log(f"mix/f1s_test/{time_to_eval}_days", self.test_f1s, on_epoch=True, add_dataloader_idx=False)
        self.log(f"mix/rcl_test/{time_to_eval}_days", self.test_rcl, on_epoch=True, add_dataloader_idx=False)

        return loss_dic


    def predict_step(self, batch_data, batch_idx, dataloader_idx=None):
        # Prepare input data
        input_dict = self.get_input_data(batch_data)

        # Forward pass through the model
        pred_lc, pred_tab, pred_mix = self.atat(**input_dict)
        pred = (
            pred_mix
            if pred_mix is not None
            else (pred_lc if pred_lc is not None else pred_tab)
        )

        if pred is None:
            raise ValueError("Invalid prediction.")

        # True labels
        y_true = batch_data["labels"].long()

        # Get predicted probabilities and class predictions
        y_pred_prob = torch.softmax(pred, dim=-1)
        y_pred = torch.argmax(y_pred_prob, dim=-1)

        # Return the required dictionary
        return {
            "id": batch_data["id"],
            "logits": pred.cpu().numpy(),
            "y_pred": y_pred.cpu().numpy(),
            "y_pred_prob": y_pred_prob.cpu().numpy(),
            "y_true": y_true.cpu().numpy(),
        }

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
        input_dict = {
            "id": batch_data["id"],
        }

        if self.use_lightcurves:
            input_dict.update(
                {
                    "data": batch_data["data"].float(),
                    "time": batch_data["time"].float(),
                    "time_alert": batch_data["time_alert"].float(),
                    "mask": batch_data["mask"].bool(),
                }
            )

        if self.use_lightcurves_err:
            input_dict.update({"data_err": batch_data["data_err"].float()})

        tabular_features = []
        mask_tabular = []
        if self.use_metadata:
            tabular_features.append(batch_data["metadata_feat"].float().unsqueeze(2))
            mask_tabular.append(batch_data["mask_metadata"].float().unsqueeze(2))

        if self.use_features:
            tabular_features.append(batch_data["extracted_feat"].float().unsqueeze(2))
            mask_tabular.append(batch_data["mask_feat"].float().unsqueeze(2))

        if tabular_features:
            input_dict["tabular_feat"] = torch.cat(tabular_features, axis=1)
            input_dict["mask_tabular"] = torch.cat(mask_tabular, axis=1)

        return input_dict

