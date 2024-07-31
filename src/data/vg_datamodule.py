import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Any, Dict, Optional, Tuple
import torch
import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.components.vg_dataset import VG_Dataset
from src.models.components.deformable_detr import DeformableDetrFeatureExtractor, DeformableDetrFeatureExtractorWithAugmentorNoCrop
from src.utils.utils import collate_fn

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
        debug=False
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

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

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
        if not self.data_train and not self.data_val and not self.data_test:
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
            self.train_dataloader = DataLoader(
                self.train_dataset,
                collate_fn=lambda x: collate_fn(x, self.feature_extractor),
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                shuffle=True,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                collate_fn=lambda x: collate_fn(x, self.feature_extractor),
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                shuffle=False
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                collate_fn=lambda x: collate_fn(x, self.feature_extractor),
                batch_size=1,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                shuffle=False
            )
            
    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader

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
    # from components.transform import UniformSampleFrames, PoseDecode, PoseCompact, Resize, RandomResizedCrop, CenterCrop, Flip, GeneratePoseTarget, FormatShape, PackActionInputs
    # train_pipeline = [
    #     UniformSampleFrames(clip_len=48),
    #     PoseDecode(),
    #     PoseCompact(hw_ratio= 1., allow_imgpad= True),
    #     Resize(scale=[-1, 64]),
    #     RandomResizedCrop(area_range= [0.56, 1.0]),
    #     Resize(scale=[56, 56], keep_ratio=False),
    #     Flip(flip_ratio=0.5, 
    #          left_kp=[1, 3, 5, 7, 9, 11, 13, 15], 
    #          right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    #     GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False),
    #     FormatShape(input_format="NCTHW_Heatmap", collapse=True),
    #     PackActionInputs(),
    # ]
    # val_pipeline = [
    #     UniformSampleFrames(clip_len=48, num_clips=1, test_mode=True),
    #     PoseDecode(),
    #     PoseCompact(hw_ratio= 1., allow_imgpad= True),
    #     Resize(scale=[-1, 64]),
    #     CenterCrop(crop_size=64),
    #     GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False),
    #     FormatShape(input_format="NCTHW_Heatmap", collapse=True),
    #     PackActionInputs(),
    # ]
    # test_pipeline = [
    #     UniformSampleFrames(clip_len=48, num_clips=10, test_mode=True),
    #     PoseDecode(),
    #     PoseCompact(hw_ratio= 1., allow_imgpad= True),
    #     Resize(scale=[-1, 64]),
    #     CenterCrop(crop_size=64),
    #     GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, 
    #                        with_limb=False, double= True, 
    #                        left_kp=[1, 3, 5, 7, 9, 11, 13, 15], 
    #                        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    #     FormatShape(input_format="NCTHW_Heatmap", collapse=True),
    #     PackActionInputs(),
    # ]
    # pipeline = {
    #     "train": train_pipeline,
    #     "val": val_pipeline,
    #     "test": test_pipeline
    # }
    
    # # data_module = NTU_Skeleton_DataModule(ann_file="/data1.local/vinhpt/dupt/self_learning/data/ntu60_2d.pkl", 
    # #                                       pipeline_transform=pipeline)
    # dataset = NTU_Skeleton_Dataset(
    #             ann_file="./data/ntu60_2d.pkl",
    #             split_mode="cross_subject",
    #             train=True,
    #             pipeline=pipeline["train"],
    #         ) 

    data_module = VG_DataModule(
        data_folder='/data/hpc/trongminh/vg', 
        feature_extractor=None,
        split="train",
        num_object_queries=100, 
        debug=False
    )
    
    import IPython; IPython.embed()
    