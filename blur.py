import torch
import cv2
import numpy as np
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
local_checkpoint_path = "sam_vit_h_4b8939.pth"

if not os.path.isfile(local_checkpoint_path):
    urllib.request.urlretrieve(checkpoint_url, local_checkpoint_path)

sam_checkpoint = local_checkpoint_path

def blur_objects(image_rgb, class_to_blur, detections, tracked_objects):
    # Initialize Segment-Anything
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    # Process tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        class_name = detections[int(obj_id)]['class_name']

        if class_name == class_to_blur:
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

            cropped_frame = image_rgb[y1:y2, x1:x2].copy()

            # Find the largest mask
            masks = mask_generator_.generate(cropped_frame)
            largest_mask = None
            largest_mask_area = 0
            for mask in masks:
                segmentation = mask["segmentation"]
                mask_area = np.sum(segmentation)
                if mask_area > largest_mask_area:
                    largest_mask_area = mask_area
                    largest_mask = mask

            if largest_mask is not None:
                # Apply blur only on the largest mask
                segmentation = largest_mask["segmentation"]
                blurred_region = cv2.GaussianBlur(cropped_frame, (99, 99), 30)
                cropped_frame = np.where(segmentation[..., None], blurred_region, cropped_frame)
                image_rgb[y1:y2, x1:x2] = cropped_frame

    return image_rgb
