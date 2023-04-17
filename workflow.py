import cv2
import sieve
from typing import Dict
from sam import SAM
from blur import blur_objects

@sieve.workflow(name="selective_object_blur")
def selective_blur_metadata_workflow(video: sieve.Video, class_to_blur: str) -> Dict:
    # Split video into frames
    images = sieve.reference("sieve-developer/video-splitter")(video)
    
    # Perform YOLOv5 object detection
    yolo_outputs = sieve.reference("sieve-developer/yolo")(images)
    
    # Perform SORT object tracking
    tracked_objects = sieve.reference("sieve-developer/sort")(yolo_outputs)

    # Get object masks using SAM
    object_masks = SAM()(images)

    # Blur objects
    blurred_images = blur_objects(images, tracked_objects, class_to_blur, object_masks)

    # Combine frames into a video
    blurred_video = sieve.reference("sieve-developer/frame-combiner")(blurred_images)

    return blurred_video, tracked_objects, object_masks