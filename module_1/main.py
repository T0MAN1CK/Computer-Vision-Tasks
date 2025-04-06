from sharpening.io_utils import load_images, save_images
from sharpening.align import align_images
from sharpening.process import resize_image, pan_sharpen


def main():
    gray, rgb_half, rgb_quarter = load_images()

    rgb_half_aligned = align_images(gray, rgb_half)
    rgb_quarter_aligned = align_images(gray, rgb_quarter)

    rgb_half_resized = resize_image(rgb_half_aligned, gray.shape)
    rgb_quarter_resized = resize_image(rgb_quarter_aligned, gray.shape)

    pan_sharp_half = pan_sharpen(gray, rgb_half_resized)
    pan_sharp_quarter = pan_sharpen(gray, rgb_quarter_resized)

    save_images(pan_sharp_half, pan_sharp_quarter)


if __name__ == "__main__":
    main()
