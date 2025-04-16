import cv2
import numpy as np
import torch


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: torch.Tensor,
    color=(0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    image_copy = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
    return image_copy
