import cv2
import numpy as np


def _binary_sobel(sobel, threshold):
    sobel = np.absolute(sobel)
    sobel = np.uint8(255 / np.max(sobel) * sobel)
    binary_grad = np.zeros_like(sobel)
    binary_grad[threshold[0] <= sobel <= threshold[1]] = 1


def abs_x_threshold(img, ksize, threshold):
    return _binary_sobel(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize), threshold)


def abs_y_threshold(img, ksize, threshold):
    return _binary_sobel(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize), threshold)


def mag_threshold(img, ksize, threshold):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = (sobel_x**2 + sobel_y**2)**(1/2)
    return _binary_sobel(sobel, threshold)


def dir_threshold(img, ksize, threshold):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_grad_dir = np.zeros_like(img)
    binary_grad_dir[threshold[0] <= sobel <= threshold[1]] = 1
    return binary_grad_dir
