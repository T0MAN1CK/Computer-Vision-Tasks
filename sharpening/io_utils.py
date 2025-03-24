import cv2
import os


def load_images():
    gray = cv2.imread("images/GRAY.JPG", cv2.IMREAD_GRAYSCALE)
    rgb_half = cv2.imread("images/RGB_half.JPG")
    rgb_quarter = cv2.imread("images/RGB_quarter.JPG")
    return gray, rgb_half, rgb_quarter


def save_images(half_img, quarter_img, output_dir="pan_sharpened_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/pan_sharpened_half.jpg", half_img)
    cv2.imwrite(f"{output_dir}/pan_sharpened_quarter.jpg", quarter_img)
    print(f"Saved sharpened images to {output_dir}/")
