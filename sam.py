import sieve
from typing import Dict, List, Tuple

@sieve.Model(
    name="sam",
    gpu=True,
    python_packages=[
        "git+https://github.com/facebookresearch/segment-anything.git",
        "torch==1.8.1",
        "torchvision==0.9.1",
        "ipython==8.4.0",
        "psutil==5.8.0",
        "seaborn==0.11.2",
        "opencv-python-headless==4.5.4.60",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    ]
)
class SAM:
    def __setup__(self):
        import os
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        # Set the path to the downloaded SAM model checkpoint
        self.sam_checkpoint_path = "sam_vit_h_4b8939.pth"

        # Check if the file exists and is not empty
        if not os.path.exists(self.sam_checkpoint_path) or os.path.getsize(self.sam_checkpoint_path) == 0:
            raise Exception("SAM checkpoint file not downloaded properly")
        
        # Load the SAM model
        self.sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint_path)
        self.sam.to(device="cuda")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.96,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
    
    def __predict__(self, img: sieve.Image) -> List:
        import cv2
        bgr_img = cv2.imread(img.path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return self.mask_generator.generate(rgb_img)
