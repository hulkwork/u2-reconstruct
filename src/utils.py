import cv2
import numpy as np


def diff_image(image1: np.ndarray, image2: np.ndarray):
    # compute difference
    difference = cv2.subtract(image1, image2)

    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(
        Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    difference[mask != 255] = [0, 0, 255]

    return difference
