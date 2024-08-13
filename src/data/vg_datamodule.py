import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Any, Dict, Optional, Tuple
import torch
import hydra
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.components.vg_dataset import VG_Dataset
from src.models.components.deformable_detr import DeformableDetrFeatureExtractor, DeformableDetrFeatureExtractorWithAugmentorNoCrop
from src.models.util import collate_fn

class VG_DataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self, 
        data_folder: str = '/data/hpc/trongminh/vg', 
        split: str = "train",
        num_object_queries=100, 
        debug=False, 
        batch_size: int = 4, 
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_folder = data_folder 
        self.split = split 
        self.num_object_queries = num_object_queries
        self.debug = debug
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
            "SenseTime/deformable-detr", size=800, max_size=1333
        )

        self.feature_extractor_train = (
            DeformableDetrFeatureExtractorWithAugmentorNoCrop.from_pretrained(
                "SenseTime/deformable-detr", size=800, max_size=1333
            )
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.setup()

    @property
    def num_classes(self):
        return len(self.train_data.coco.cats)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.train_dataset = VG_Dataset(
                data_folder=self.data_folder,
                feature_extractor=self.feature_extractor_train,
                split="train",
                num_object_queries=self.num_object_queries,
                debug=self.debug,
            )
            self. val_dataset = VG_Dataset(
                data_folder=self.data_folder,
                feature_extractor=self.feature_extractor,
                split="val",
                num_object_queries=self.num_object_queries,
            )
            self.test_dataset = VG_Dataset(
                data_folder=self.data_folder,
                feature_extractor=self.feature_extractor,
                split=self.split,
                num_object_queries=self.num_object_queries,
            )
            self.train_dataloader_object = DataLoader(
                self.train_dataset,
                collate_fn=lambda x: collate_fn(x, self.feature_extractor),
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                shuffle=True,
            )
            self.val_dataloader_object = DataLoader(
                self.val_dataset,
                collate_fn=lambda x: collate_fn(x, self.feature_extractor),
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                shuffle=False
            )
            self.test_dataloader_object = DataLoader(
                self.test_dataset,
                collate_fn=lambda x: collate_fn(x, self.feature_extractor),
                batch_size=1,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                shuffle=False
            )
            
    def train_dataloader(self):
        return self.train_dataloader_object

    def val_dataloader(self):
        return self.val_dataloader_object

    def test_dataloader(self):
        return self.test_dataloader_object

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":

    feature_extractor_train = (
        DeformableDetrFeatureExtractorWithAugmentorNoCrop.from_pretrained(
            "SenseTime/deformable-detr", size=800, max_size=1333
        )
    )

    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        "SenseTime/deformable-detr", size=800, max_size=1333
    )

    train_dataset = VG_Dataset(
        data_folder='/data/hpc/trongminh/visual_genome',
        feature_extractor=feature_extractor_train,
        split="train",
        num_object_queries=100, 
        debug=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=4,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        shuffle=True,
    )

    val_dataset = VG_Dataset(
        data_folder='/data/hpc/trongminh/visual_genome',
        feature_extractor=feature_extractor,
        split="val",
        num_object_queries=100,
    )
    test_dataset = VG_Dataset(
        data_folder='/data/hpc/trongminh/visual_genome',
        feature_extractor=feature_extractor,
        split="test",
        num_object_queries=100,
    )
    
    import IPython; IPython.embed()
    