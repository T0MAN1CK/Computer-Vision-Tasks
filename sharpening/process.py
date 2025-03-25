import cv2
import numpy as np


def resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Resizes the given image to the specified (height, width) shape.
    """
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def pan_sharpen(gray: np.ndarray, rgb_aligned: np.ndarray) -> np.ndarray:
    """
    Injects high-res gray detail into the RGB image using HSV space pan-sharpening.
    """
    hsv = cv2.cvtColor(rgb_aligned, cv2.COLOR_BGR2HSV)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = gray_norm
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
