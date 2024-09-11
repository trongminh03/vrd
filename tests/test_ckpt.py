import torch
import hydra

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

with hydra.initialize(version_base="1.3", config_path="../configs", ):
    cfg = hydra.compose(config_name='train.yaml')
    print(cfg)

egtr_ckpt_default = "/workspace/vrd/ckpt/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/best.ckpt"
egtr_ckpt_retrain = "/workspace/vrd/ckpt/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/retrain/checkpoints/epoch=00-validation_loss=1.71.ckpt"
egtr_default = torch.load(egtr_ckpt_default, map_location="cpu") 
egtr_retrain = torch.load(egtr_ckpt_retrain, map_location="cpu")

pretrained = "/workspace/vrd/detection_models/pretrained_detr__SenseTime__deformable-detr/batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune/version_0/checkpoints/epoch=26-validation_loss=10.31.ckpt"
detr = torch.load(pretrained, map_location="cpu")

print(len(egtr_default['state_dict'].keys()))
print(len(egtr_retrain['state_dict'].keys()))
print(len(detr['state_dict'].keys())) 

set(egtr_default['state_dict']) - set(egtr_retrain['state_dict'])

import IPython; IPython.embed()