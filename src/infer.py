from glob import glob
import torch
from PIL import Image
import numpy as np
import json
from torchvision.ops import box_convert
import cv2
import os
from tqdm import tqdm
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.box_ops import rescale_bboxes
from src.utils.pytorch_misc import argsort_desc
from src.models.components.egtr import DetrForSceneGraphGeneration
from src.models.components.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor

# config
multiple_sgg_evaluator = False
architecture = "SenseTime/deformable-detr"
min_size = 800
max_size = 1333
max_topk = 10
artifact_path = "/workspace/vrd/ckpt/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/"

# label 
data_path = "/workspace/data/visual_genome"
with open(f"{data_path}/rel.json", "r") as f: 
    rel = json.load(f)
    rel_labels = rel["rel_categories"][1:]  # remove 'no_relation' category

with open(f"{data_path}/val.json", "r") as f: 
    val = json.load(f)
    obj_categories = val["categories"]
    obj_labels = [item['name'] for item in obj_categories]

# feature extractor
feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    architecture, size=min_size, max_size=max_size
)

# folder containing images
img_folder_path = "/workspace/vrd/tests/imgs/"
output_folder = "/workspace/vrd/tests/outputs/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load model config and checkpoint
config = DeformableDetrConfig.from_pretrained(artifact_path)
model = DetrForSceneGraphGeneration.from_pretrained(
    architecture, config=config, ignore_mismatched_sizes=True
)
ckpt_path = sorted(
    glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
)[-1]
state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
for k in list(state_dict.keys()):
    state_dict[k[6:]] = state_dict.pop(k)  # "model."

model.load_state_dict(state_dict)
model.cuda()
model.eval()

# process each image in the folder
img_paths = glob(f"{img_folder_path}/*.jpg")  # Adjust extension if necessary
for img_path in tqdm(img_paths, desc="Processing images"):
    # Load and process image
    image = Image.open(img_path)
    image_tensor = feature_extractor(image, return_tensors="pt")

    # output
    outputs = model(
        pixel_values=image_tensor['pixel_values'].cuda(), 
        pixel_mask=image_tensor['pixel_mask'].cuda(), 
        output_attention_states=True
    )

    pred_logits = outputs['logits'][0]
    obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), -1)
    pred_boxes = outputs['pred_boxes'][0]

    sub_ob_scores = torch.outer(obj_scores, obj_scores)
    sub_ob_scores[torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))] = 0.0  # prevent self-connection

    pred_connectivity = outputs['pred_connectivity'][0]
    pred_rel = outputs['pred_rel'][0]
    pred_rel = torch.mul(pred_rel, pred_connectivity)

    if multiple_sgg_evaluator: 
        triplet_scores = torch.mul(pred_rel, sub_ob_scores.unsqueeze(-1))
        pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().detach().numpy())[:max_topk, :]
        rel_scores = pred_rel.cpu().clone().detach().numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
        pred_rels = pred_rel_inds
        predicate_scores = rel_scores
    else:  # single_sgg_evaluator
        triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)
        pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().detach().numpy())[:max_topk, :]
        rel_scores = pred_rel.cpu().clone().detach().numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1]]

        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1)))
        predicate_scores = rel_scores.max(1)

    subject_classes = pred_classes[pred_rels[:, 0]].cpu().numpy()
    object_classes = pred_classes[pred_rels[:, 1]].cpu().numpy()

    pred_rels_classes = np.column_stack((subject_classes, object_classes, pred_rels[:, 2]))

    # Read img
    test_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    height, width, _ = test_img.shape
    orig_size = torch.tensor([height, width])

    pred_boxes = rescale_bboxes(pred_boxes.cpu(), torch.flip(orig_size, dims=[0]))

    overlay = test_img.copy()  # Create a copy of the original image for overlay
    for i in range(0, max_topk):
        s_ind, o_ind, p = pred_rels[i]
        s = pred_classes[s_ind]
        o = pred_classes[o_ind]

        s_label = obj_labels[s]
        o_label = obj_labels[o]
        p_label = rel_labels[p]

        print(f'<{s_label} -- {p_label} -- {o_label}>')

        s_box = pred_boxes[s_ind]
        o_box = pred_boxes[o_ind]

        color = (np.random.uniform(0, 255), np.random.uniform(0, 255), np.random.uniform(0, 255))

        # Draw line and labels
        s_center = (int((s_box[0] + s_box[2]) / 2), int((s_box[1] + s_box[3]) / 2))
        o_center = (int((o_box[0] + o_box[2]) / 2), int((o_box[1] + o_box[3]) / 2))

        cv2.putText(overlay, s_label, s_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(overlay, o_label, o_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.line(overlay, s_center, o_center, color, 2)

        mid_point = (int((s_center[0] + o_center[0]) / 2), int((s_center[1] + o_center[1]) / 2))
        cv2.putText(overlay, p_label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Blend overlay onto original image
        opacity = 0.6
        cv2.addWeighted(overlay, opacity, test_img, 1 - opacity, 0, test_img)

    # Save the image with predictions
    output_img_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_img_path, test_img)
