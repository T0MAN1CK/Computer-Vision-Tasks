import cv2


def resize_image(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def pan_sharpen(gray, rgb_aligned):
    hsv = cv2.cvtColor(rgb_aligned, cv2.COLOR_BGR2HSV)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = gray_norm
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
