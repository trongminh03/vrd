from glob import glob
import torch
from PIL import Image

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.egtr import DetrForSceneGraphGeneration
from src.models.components.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor

# config
architecture = "SenseTime/deformable-detr"
min_size = 800
max_size = 1333
artifact_path = "/workspace/vrd/ckpt/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/"

# feature extractor
feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    architecture, size=min_size, max_size=max_size
)

# inference image
image = Image.open("/workspace/vrd/tests/imgs/test.jpg")
image = feature_extractor(image, return_tensors="pt")

# model
config = DeformableDetrConfig.from_pretrained(artifact_path)
model = DetrForSceneGraphGeneration.from_pretrained(
    architecture, config=config, ignore_mismatched_sizes=True
)
ckpt_path = sorted(
    glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
)[-1]
state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
import IPython; IPython.embed()
for k in list(state_dict.keys()):
    state_dict[k[6:]] = state_dict.pop(k)  # "model."

model.load_state_dict(state_dict)
model.cuda()
model.eval()

# output
outputs = model(
    pixel_values=image['pixel_values'].cuda(), 
    pixel_mask=image['pixel_mask'].cuda(), 
    output_attention_states=True
)

pred_logits = outputs['logits'][0]
obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), -1)
pred_boxes = outputs['pred_boxes'][0]

pred_connectivity = outputs['pred_connectivity'][0]
pred_rel = outputs['pred_rel'][0]
pred_rel = torch.mul(pred_rel, pred_connectivity)

# get valid objects and triplets
obj_threshold = 0.3
valid_obj_indices = (obj_scores >= obj_threshold).nonzero()[:, 0]

valid_obj_classes = pred_classes[valid_obj_indices] # [num_valid_objects]
valid_obj_boxes = pred_boxes[valid_obj_indices] # [num_valid_objects, 4]

rel_threshold = 1e-4
valid_triplets = (pred_rel[valid_obj_indices][:, valid_obj_indices] >= rel_threshold).nonzero() # [num_valid_triplets, 3]

import IPython; IPython.embed()