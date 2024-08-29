from functools import partial
import json
import numpy as np
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


from src.models.components.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrFeatureExtractorWithAugmentorNoCrop,
)
from src.models.components.egtr import DetrForSceneGraphGeneration

class EGTRLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        from_scratch: bool, 
        log_print=False,
        config_params=None,
        # id2label,
        # rel_categories,
        # multiple_sgg_evaluator,
        # multiple_sgg_evaluator_list,
        # single_sgg_evaluator,
        # single_sgg_evaluator_list,
        # coco_evaluator,
        # feature_extractor,
        # fg_matrix,
        num_classes=150
    ) -> None:
        """Initialize a `EGTRLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.from_scratch = from_scratch
        self.log_print = log_print

        # Load config
        self.config = DeformableDetrConfig.from_pretrained(config_params.pretrained)
        for k, v in config_params.items():
            setattr(self.config, k, v)
        
        self.config.num_rel_labels = len(self.config.rel_categories)
        self.config.num_labels = max(self.config.id2label.keys()) + 1

        fg_matrix = np.array(json.loads(self.config.fg_matrix))

        if self.from_scratch: 
            self.net = net(config=self.config, fg_matrix=fg_matrix)
            self.net.model.backbone.load_state_dict(
                torch.load(f"{self.config.backbone_dirpath}/{self.config.backbone}.pt")
            )
            self.initialized_keys = []
        else: 
            net = net(config=self.config)
            self.net, load_info = net.from_pretrained(
                self.config.pretrained,
                config=self.config,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
                fg_matrix=fg_matrix,
            )
            # self.net, load_info = DetrForSceneGraphGeneration.from_pretrained(
            #     self.config.pretrained,
            #     config=self.config,
            #     ignore_mismatched_sizes=True,
            #     output_loading_info=True,
            #     fg_matrix=fg_matrix,
            # )

            self.initialized_keys = load_info["missing_keys"] + [
                _key for _key, _, _ in load_info["mismatched_keys"]
            ]

        # if self.config.main_trained:
        #     state_dict = torch.load(main_trained, map_location="cpu")["state_dict"]
        #     for k in list(state_dict.keys()):
        #         state_dict[k[6:]] = state_dict.pop(k)  # "model."
        #     self.model.load_state_dict(state_dict, strict=False)

        # self.multiple_sgg_evaluator = multiple_sgg_evaluator 
        # self.multiple_sgg_evaluator_list = multiple_sgg_evaluator_list 
        # self.single_sgg_evaluator = single_sgg_evaluator
        # self.single_sgg_evaluator_list = single_sgg_evaluator_list
        # self.coco_evaluator = coco_evaluator
        # self.feature_extractor = feature_extractor

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        pixel_val = x[0] 
        pixel_mask = x[1]
        return self.net(pixel_val, pixel_mask, output_attentions=False,
                        output_attention_states=True,
                        output_hidden_states=True,)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        if self.log_print:
            print("START TRAINING")

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pixel_val = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        x = (pixel_val, pixel_mask)
        y = batch['labels']
        # output = self.forward(x) 
        # loss = self.criterion(logits, y) 
        # preds = torch.argmax(logits, dim=1)
        # return loss, preds, y
        outputs = self.net(pixel_val, pixel_mask, labels=y, output_attentions=False, output_attention_states=True, output_hidden_states=True)
        # import IPython; IPython.embed()
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        del outputs
        return loss, loss_dict

    # def training_step(
    #     self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    # ) -> torch.Tensor:
    #     """Perform a single training step on a batch of data from the training set.

    #     :param batch: A batch of data (a tuple) containing the input tensor of images and target
    #         labels.
    #     :param batch_idx: The index of the current batch.
    #     :return: A tensor of losses between model predictions and targets.
    #     """
    #     loss, preds, targets = self.model_step(batch)

    #     # update and log metrics
    #     self.train_loss(loss)
    #     self.train_acc(preds, targets)
    #     self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

    #     # return loss or backpropagation will fail
    #     return loss 

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.model_step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "training_loss": loss.item(),
        }
        log_dict.update({f"training_{k}": v.item() for k, v in loss_dict.items()})
        self.log_dict(log_dict)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    # def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
    #     """Perform a single validation step on a batch of data from the validation set.

    #     :param batch: A batch of data (a tuple) containing the input tensor of images and target
    #         labels.
    #     :param batch_idx: The index of the current batch.
    #     """
    #     loss, preds, targets = self.model_step(batch)

    #     # update and log metrics
    #     self.val_loss(loss)
    #     self.val_acc(preds, targets)
    #     self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
 
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.model_step(batch)
        loss_dict["loss"] = loss
        del loss
        return loss_dict

    # def on_validation_epoch_end(self) -> None:
    #     "Lightning hook that is called when a validation epoch ends."
    #     acc = self.val_acc.compute()  # get current val acc
    #     self.val_acc_best(acc)  # update best so far val acc
    #     # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
    #     # otherwise metric would be reset by lightning after each epoch
    #     self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
    #     if self.log_print:
    #         print(f"Epoch {self.current_epoch} end, Train/loss: {round(float(self.train_loss.compute()), 3)}, Train/acc: {round(float(self.train_acc.compute()), 4)}, Val/loss: {round(float(self.val_loss.compute()), 3)}, Val/acc: {round(float(self.val_acc.compute()), 4)}, Val/acc_best: {round(float(self.val_acc_best.compute()), 4)}")

    def on_validation_epoch_end(self, outputs):
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "epoch": torch.tensor(self.current_epoch, dtype=torch.float32),
        }
        for k in outputs[0].keys():
            log_dict[f"validation_" + k] = (
                torch.stack([x[k] for x in outputs]).mean().item()
            )
        self.log_dict(log_dict, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = EGTRLitModule(None, None, None, None)
