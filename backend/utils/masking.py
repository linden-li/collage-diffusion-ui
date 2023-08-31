import numpy as np
from typing import Dict
from PIL import Image
import matplotlib.path as mpath

def generate_mask(
    clicks: Dict[str, float],
    img_width: int, 
    img_height: int
) -> np.ndarray:
    if len(clicks) == 0:
        # If no clicks, return a mask of all 1s
        return np.ones((img_height, img_width), dtype=np.uint8)

    polygon = mpath.Path(
        [(click['x'] * img_width, click['y'] * img_height) for click in clicks]
    )
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for i in range(img_height):
        for j in range(img_width):
            if polygon.contains_point((j, i)):
                mask[i, j] = 1
    return mask

def compute_bounding_box(mask: np.ndarray) -> tuple:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

def apply_mask(
    img: Image, 
    mask: np.ndarray
) -> Image: 
    img = np.array(img)
    img[:, :, 3] = img[:, :, 3] * mask
    
    # Compute bounding box
    xmin, ymin, xmax, ymax = compute_bounding_box(mask)
    
    # Crop image using bounding box
    cropped_img = img[ymin:ymax+1, xmin:xmax+1, :]
    
    return Image.fromarray(cropped_img)