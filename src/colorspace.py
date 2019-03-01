import cv2


def rgb_channels(img):
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def hsl_channels(img):
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hsl[:, :, 0], hsl[:, :, 2], hsl[:, :, 1]


def hsv_channels(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
