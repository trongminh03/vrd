import hydra
from glob import glob
import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from tqdm import tqdm

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from src.models.util import collate_fn
from src.data.components.vg_dataset import VG_Dataset, vg_get_statistics
from src.evaluation.coco_eval import CocoEvaluator, evaluate
from src.evaluation.evaluate_egtr import calculate_fps
from src.evaluation.vg_eval import BasicSceneGraphEvaluator
from src.models.components.egtr import DetrForSceneGraphGeneration

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig): 
    # Load datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # config model param
    train_dataset = datamodule.train_dataset
    cats = train_dataset.coco.cats
    id2label = {k - 1: v["name"] for k, v in cats.items()}
    
    fg_matrix = vg_get_statistics(train_dataset, must_overlap=True)
    fg_matrix_json = json.dumps(fg_matrix.tolist())
    
    rel_categories = train_dataset.rel_categories

    # update model config
    cfg.model.config_params.id2label = id2label
    cfg.model.config_params.rel_categories = rel_categories
    cfg.model.config_params.fg_matrix = fg_matrix_json

    # Load model
    model: LightningModule = hydra.utils.instantiate(cfg.model) 
    if "ckpt_path" in cfg:
        state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
    
    test_dataloader = datamodule.test_dataloader()
    model.eval() 
    for batch in tqdm(test_dataloader): 
        pixel_values=batch["pixel_values"].cuda()
        pixel_mask=batch["pixel_mask"].cuda()
        y = batch['labels']
        targets = [{k: v.cpu() for k, v in label.items()} for label in batch["labels"]] 
        x = (pixel_values, pixel_mask)
        output = model(x) 
        import IPython; IPython.embed()
        


    import IPython; IPython.embed()

if __name__ == "__main__":
    main()


# if __name__ == '__main__':
#     def str2bool(v):
#         if isinstance(v, bool):
#             return v
#         if v.lower() in ("yes", "true", "t", "y", "1"):
#             return True
#         elif v.lower() in ("no", "false", "f", "n", "0"):
#             return False
#         else:
#             raise argparse.ArgumentTypeError("Boolean value expected.")

#     parser = argparse.ArgumentParser()
#     # Path
#     parser.add_argument("--data_path", type=str, default="/data/hpc/trongminh/visual_genome/")
#     parser.add_argument(
#         "--artifact_path",
#         type=str,
#         required=True,
#     )
#     # Architecture
#     parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
#     parser.add_argument("--num_queries", type=int, default=200)

#     # Evaluation
#     parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
#     parser.add_argument("--eval_batch_size", type=int, default=1)
#     parser.add_argument("--eval_single_preds", type=str2bool, default=True)
#     parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)

#     parser.add_argument("--logit_adjustment", type=str2bool, default=False)
#     parser.add_argument("--logit_adj_tau", type=float, default=0.3)

#     # FPS
#     parser.add_argument("--min_size", type=int, default=800)
#     parser.add_argument("--max_size", type=int, default=1333)
#     parser.add_argument("--infer_only", type=str2bool, default=False)

#     # Speed up
#     parser.add_argument("--num_workers", type=int, default=4)
#     args, unknown = parser.parse_known_args()  # to ignore args when training

#     # Feature extractor
#     feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
#         args.architecture, size=args.min_size, max_size=args.max_size
#     )

#     # Dataset
#     if "visual_genome" in args.data_path:
#         test_dataset = VG_Dataset(
#             data_folder=args.data_path,
#             feature_extractor=feature_extractor,
#             split=args.split,
#             num_object_queries=args.num_queries,
#         )
#     id2label = {
#             k - 1: v["name"] for k, v in test_dataset.coco.cats.items()
#         }  # 0 ~ 149
#     coco_evaluator = CocoEvaluator(
#         test_dataset.coco, ["bbox"]
#     )  # initialize evaluator with ground truths
#     oi_evaluator = None

#     # Dataloader
#     test_dataloader = DataLoader(
#         test_dataset,
#         collate_fn=lambda x: collate_fn(x, feature_extractor),
#         batch_size=args.eval_batch_size,
#         pin_memory=True,
#         num_workers=args.num_workers,
#         persistent_workers=True,
#     )

#     # Evaluator
#     multiple_sgg_evaluator = None
#     single_sgg_evaluator = None
#     if args.eval_multiple_preds:
#         multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
#     if args.eval_single_preds:
#         single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)

#     # Model
#     config = DeformableDetrConfig.from_pretrained(args.artifact_path)
#     config.logit_adjustment = args.logit_adjustment
#     config.logit_adj_tau = args.logit_adj_tau

#     model = DetrForSceneGraphGeneration.from_pretrained(
#         args.architecture, config=config, ignore_mismatched_sizes=True
#     )
#     ckpt_path = sorted(
#         glob(f"{args.artifact_path}/checkpoints/epoch=*.ckpt"),
#         key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
#     )[-1]
#     state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
#     for k in list(state_dict.keys()):
#         state_dict[k[6:]] = state_dict.pop(k)  # "model."

#     model.load_state_dict(state_dict)
#     model.cuda()
#     model.eval()

#     # FPS
#     if args.infer_only:
#         calculate_fps(model, test_dataloader)
#     # Eval
#     else:
#         metric = evaluate(
#             model,
#             test_dataloader,
#             max(id2label.keys()) + 1,
#             multiple_sgg_evaluator,
#             single_sgg_evaluator,
#             oi_evaluator,
#             coco_evaluator,
#             feature_extractor,
#         )

#         # Save eval metric
#         device = "".join(torch.cuda.get_device_name(0).split()[1:2])
#         filename = f'{ckpt_path.replace(".ckpt", "")}__{args.split}__{len(test_dataloader)}__{device}'
#         if args.logit_adjustment:
#             filename += f"__la_{args.logit_adj_tau}"
#         metric["eval_arg"] = args.__dict__
#         with open(f"{filename}.json", "w") as f:
#             json.dump(metric, f)
#         print("metric is saved in", f"{filename}.json")