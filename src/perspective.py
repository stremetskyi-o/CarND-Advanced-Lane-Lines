import cv2
import numpy as np


class PerspectiveTransform:
    _y_start_src = 450
    _y_end_src = 670
    _default_src = ((267, _y_end_src), (1040, _y_end_src), (687, _y_start_src), (594, _y_start_src))
    _default_shape = (720, 1280)

    def __init__(self, src=_default_src, shape=_default_shape):
        """
        :param src : 4 points in counter-clockwise order, starting from bottom-left
        :param shape : shape of the image being transformed
        """
        self.src = np.float32(src)
        self.dst = np.float32([[self.src[0, 0], shape[0]],
                               [self.src[1, 0], shape[0]],
                               [self.src[1, 0], 0],
                               [self.src[0, 0], 0]])
        self.m = cv2.getPerspectiveTransform(self.src, self.dst)

    def draw_src(self, img):
        cv2.polylines(img, [self.src.astype(np.int32)], True, (0, 255, 0), 2)

    def draw_dst(self, img):
        cv2.polylines(img, [self.dst.astype(np.int32)], True, (0, 255, 0), 2)

    def warp(self, img):
        return cv2.warpPerspective(img, self.m, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.m, img.shape[1::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
