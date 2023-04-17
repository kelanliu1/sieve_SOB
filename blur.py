import sieve
from typing import List, Dict

@sieve.function(
    name="blur_objects",
    gpu=False,
    python_version="3.8",
    iterator_input=True,
    python_packages=[
        'uuid==1.30'
    ]
)
def blur_objects(images: sieve.Image, tracked_objects: List[Dict[str, List[Dict]]], class_to_blur: str, object_masks: List) -> sieve.Image:
    import numpy as np
    import cv2
    import uuid

    image_paths = []
    for im in images:
        image_paths.append(im)

    tracked_objects_by_frame_number = {}
    for obj_dict in tracked_objects:
        for key, value in obj_dict.items():
            frame_number = value[0]['frame_number']
            tracked_objects_by_frame_number[frame_number] = value

    images_by_frame_number = sorted(image_paths, key=lambda k: k.frame_number)
    object_masks_by_frame_number = sorted(object_masks, key=lambda k: k[0]['frame_number'])

    for i in range(len(images_by_frame_number)):
        image = images_by_frame_number[i]
        img = cv2.imread(image.path)
        objects_in_frame = tracked_objects_by_frame_number.get(i, [])
        masks_in_frame = object_masks_by_frame_number[i]

        # Find the largest mask in the bounding box
        largest_area = 0
        largest_mask = None
        for obj, mask in zip(objects_in_frame, masks_in_frame):
            if obj['class'] == class_to_blur:
                mask_area = mask['area']
                if mask_area > largest_area:
                    largest_area = mask_area
                    largest_mask = mask

        # If a mask is found, blur it
        if largest_mask is not None:
            mask_array = largest_mask['segmentation']
            blurred_img = cv2.GaussianBlur(img, (99, 99), 30)

            # Replace the original image with the blurred version within the mask
            img[mask_array] = blurred_img[mask_array]

        new_path = f'{uuid.uuid4()}.jpg'
        cv2.imwrite(new_path, img)
        yield sieve.Image(path=new_path, frame_number=obj['frame_number'], fps=image.fps)
