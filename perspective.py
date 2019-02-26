import cv2
import numpy as np

_x_start = 267
_x_end = 1040
_y_end_src = 670
_y_start_src = 450
_y_start_dst = 0
_y_end_dst = 720
_src = np.float32([[_x_start, _y_end_src], [_x_end, _y_end_src], [687, _y_start_src], [594, _y_start_src]])
_dst = np.float32([[_x_start, _y_end_dst], [_x_end, _y_end_dst], [_x_end, _y_start_dst], [_x_start, _y_start_dst]])


def get_perspective_transform():
    return cv2.getPerspectiveTransform(_src, _dst)


def draw_src(img):
    cv2.polylines(img, [_src.astype(np.int32)], True, (0, 255, 0), 2)


def draw_dst(img):
    cv2.polylines(img, [_dst.astype(np.int32)], True, (0, 255, 0), 2)


def warp(img, m):
    return cv2.warpPerspective(img, m, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def unwarp(img, m):
    return cv2.warpPerspective(img, m, img.shape[1::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
