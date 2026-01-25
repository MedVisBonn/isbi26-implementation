"""
- PAPER
https://arxiv.org/pdf/2008.07357.pdf
- CODE
https://github.com/kechua/DART20/blob/master/damri/model/unet.py
"""
from __future__ import annotations

from typing import (
    Dict,
    List,
)
from datetime import datetime
import re
from omegaconf import OmegaConf
import wandb
import torch
from torch import (
    nn,
    log, 
    tensor
)
import time
from torch.special import entr
from torch.nn.functional import one_hot
from monai.networks.nets import UNet
from monai.networks.layers.factories import Norm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.metrics.surface_dice import compute_surface_dice
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import MulticlassCalibrationError, BinaryCalibrationError
import surface_distance as sd

from utils import build_checkpoint_filename



def get_unet_module_trainer(
    data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    trainer_cfg: OmegaConf
):
    # infered variables
    patience = model_cfg.patience * 2
    now = datetime.now()
    # Strict naming via helper: DATASET_SPLIT_DROPOUT_DATE
    filename = build_checkpoint_filename(data_cfg, model_cfg, now)

    # init logger
    if trainer_cfg.logging:
        wandb.finish()
        logger = WandbLogger(
            project="MIDL25", 
            log_model=True, 
            name=filename
        )
    else:
        logger = None

    # return trainer
    return L.Trainer(
        limit_train_batches=trainer_cfg.limit_train_batches,
        max_epochs=trainer_cfg.max_epochs,
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor=trainer_cfg.early_stopping.monitor, 
                mode=trainer_cfg.early_stopping.mode, 
                patience=patience
            ),
            ModelCheckpoint(
                dirpath=trainer_cfg.model_checkpoint.dirpath,
                filename=filename,
                save_top_k=trainer_cfg.model_checkpoint.save_top_k, 
                monitor=trainer_cfg.model_checkpoint.monitor,
            )
        ],
        precision='16-mixed',
        gradient_clip_val=0.5,
        devices=[0],
        limit_test_batches=50
    )



def get_unet_module(
    cfg: OmegaConf,
    metadata: Dict,
    load_from_checkpoint: bool = False
):
    unet = UNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        channels=[cfg.n_filters_init * 2 ** i for i in range(cfg.depth)],
        strides=[2] * (cfg.depth - 1),
        num_res_units=4,
        norm=Norm.INSTANCE,
        dropout=cfg.dropout,
    )
    if load_from_checkpoint:
        return LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            model=unet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            map_location='cuda:0'
        )
    else:
        return LightningSegmentationModel(
            model=unet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            metadata=metadata
        )


class LightningSegmentationModel(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        patience: int = 5,
        binary_target: bool = False,
        metadata: Dict[str, OmegaConf] = None
    ):
        super().__init__()
        # this would save the model as hyperparameter, not desired!
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.patience = patience
        self.metadata = metadata
        self.loss = DiceCELoss(
            softmax=False if binary_target else True,
            sigmoid=True if binary_target else False,
            to_onehot_y=False if binary_target else True,
        )
        # self.val_loss = DiceCELoss(
        #     softmax=False if binary_target else True,
        #     sigmoid=True if binary_target else False,
        #     to_onehot_y=False if binary_target else True,
        #     reduction='none'
        # )
        self.dsc = DiceMetric(include_background=False, reduction="none")
        # self.hausdorff = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
        if self.metadata['unet']['out_channels'] == 1:
            num_classes = 2
        else:
            num_classes = self.metadata['unet']['out_channels']
        self.sdsc = SurfaceDiceMetric(include_background=False, class_thresholds=(num_classes-1)*[3], reduction="none")

        self.binary_ece = BinaryCalibrationError(
            n_bins=15,
            norm="l1",
            ignore_index=None,   # set to 0/1 only if you truly want to drop a label value
        )
        self.multiclass_ece = MulticlassCalibrationError(
            num_classes=num_classes,
            n_bins=15,
            norm="l1",
            ignore_index=0,      # drop background pixels from ECE
            validate_args=False, # optional speed
        )


    def forward(self, inputs):        
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean()
        self.dsc.reset()
        self.log_dict({
            'train_loss': loss,
            'train_dsc': dsc,
        })
        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean().nan_to_num(0)
        self.dsc.reset()
        self.log_dict({
            'val_loss': loss,
            'val_dsc': dsc,
        })
        return {
            'loss': loss,
        }
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean().nan_to_num(0)
        self.dsc.reset()

        self.log_dict({
            'test_loss': loss,
            'test_dsc': dsc,
        })
        return {
            'loss': loss,
            'dsc': dsc
        }
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        loss = self.loss(outputs, target)
        # print(loss.shape)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            probs = torch.softmax(outputs, 1).detach()
            outputs = outputs.argmax(dim=1, keepdim=True).detach()
            ece = self.multiclass_ece(probs, target).detach().nanmean(-1).nan_to_num(0).cpu()

        else:
            probs = outputs.sigmoid().detach()
            outputs = (outputs > 0).long().detach()
            ece = self.binary_ece(probs, target).detach().nanmean(-1).nan_to_num(0).cpu()

        predicted_segmentation = one_hot(outputs.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
        target_segmentation = one_hot(target.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
        dice = DiceMetric(include_background=False, reduction="none")(
            predicted_segmentation, target_segmentation
        ).detach().nanmean(-1).nan_to_num(0).cpu()
        
        # surface dice had a memory leak related to CuCim, which has been fixed in the latest version
        surface_dice = self.sdsc(predicted_segmentation, target_segmentation).detach().nanmean(-1).nan_to_num(0).cpu()
        # self.dsc.reset()
        

        entropy = 1 - (entr(probs).sum(1).mean((-1, -2)) / log(tensor(num_classes)))

        metrics = {
            'dice': dice,
            'loss': loss.cpu().detach(),
            # 'hausdorff': hausdorff,
            'surface_dice': surface_dice,
            'entropy': entropy.cpu().detach(),
            'ece': ece
        }

        return metrics
                

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }