import sieve
from typing import Dict
from blur import blur_objects
import cv2

@sieve.workflow(name="selective_blur")
def selective_blur_workflow(video: sieve.Video, class_to_blur: str) -> Dict:
    images = sieve.reference("sieve-developer/video-splitter")(video)
    yolo_outputs = sieve.reference("sieve-developer/yolo")(images)
    sort_outputs = sieve.reference("sieve-developer/sort")(yolo_outputs)

    blurred_images = []
    for img, detections, tracked_objects in zip(images, yolo_outputs, sort_outputs):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blurred_rgb = blur_objects(image_rgb, class_to_blur, detections, tracked_objects)
        blurred_image = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)
        blurred_images.append(blurred_image)

    # Combine the images back into a video, or process them as desired
    # ...

    return {
        # Your output
    }