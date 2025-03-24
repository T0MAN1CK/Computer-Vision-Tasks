import cv2
import numpy as np


def align_images(gray, rgb):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray, None)
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(rgb_gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(des1, des2, None))
    matches.sort(key=lambda x: x.distance)
    matches = matches[: int(len(matches) * 0.1)]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    aligned = cv2.warpPerspective(rgb, H, (gray.shape[1], gray.shape[0]))
    return aligned
