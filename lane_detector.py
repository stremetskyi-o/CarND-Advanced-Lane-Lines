import cv2
import numpy as np

import calibration_params
import colorspace
import gradient
from perspective import PerspectiveTransform

lane_width_m = 3.7
lane_width_p = 1026
lane_dash_length_m = 3.0
lane_dash_length_p = 77


class LaneDetector:

    def __init__(self, window_width=100, window_height=80, draw_convolution_windows=False, hist_height=50,
                 overlay_lanes=True):
        self.window_width = window_width
        self.window_height = window_height
        self.draw_convolution_windows = draw_convolution_windows & (not overlay_lanes)
        self.hist_height = hist_height
        self.overlay_lanes = overlay_lanes

        self.calibration_params = calibration_params.load()
        self.perspective = PerspectiveTransform()

        self.m2p_ratio_x = lane_width_m / lane_width_p
        self.m2p_ratio_y = lane_dash_length_m / lane_dash_length_p

    def process_image(self, img):
        # Prepare image
        img = cv2.undistort(img, *self.calibration_params)
        warped = self.perspective.warp(img)

        # Separate channels
        rgb_r = colorspace.rgb_channels(warped)[0]
        hsl_s = colorspace.hsl_channels(warped)[1]

        # Calculate and combine gradients
        r_x = gradient.abs_x(rgb_r, 9, (40, 170))
        s_x = gradient.abs_x(hsl_s, 9, (25, 220))
        lines_img = np.zeros_like(r_x)
        lines_img[(r_x == 1) | (s_x == 1)] = 1

        # Calculate histogram to find start of the lanes at bottom of the image
        hist = np.sum(lines_img[lines_img.shape[0] * 2 // 3:], axis=0)
        # Multiplication and clipping allows to correctly detect dashed lines
        hist = np.clip(hist * 3, 0, np.max(hist))

        line_centers = self.find_centers(hist, lines_img, warped)
        lines_img = self.highlight_features(lines_img, line_centers)

        if self.overlay_lanes:
            lines_img = self.perspective.unwarp(lines_img)
            img = cv2.addWeighted(img, 0.8, lines_img, 1, 0)
            self.draw_hist(img, hist)
            return img

        return lines_img

    def find_centers(self, hist, lines_img, warped):
        window = np.ones(self.window_width)
        window_width2 = self.window_width // 2
        lines_img_width2 = lines_img.shape[1] // 2

        # Find initial left and right lines pixels using convolution
        l_line = np.argmax(np.convolve(hist[:lines_img_width2], window)) - window_width2
        r_line = lines_img_width2 + np.argmax(np.convolve(hist[lines_img_width2:], window)) - window_width2
        # TODO: Check if there is enough distance between lines
        # TODO: Check scenario where there is only 1 line
        line_centers = [(l_line, r_line)]
        for n in range(1, warped.shape[0] // self.window_height):
            slice_bottom = lines_img.shape[0] - self.window_height * n
            slice_hist = np.sum(lines_img[slice_bottom - self.window_height:slice_bottom], axis=0)
            l_line = LaneDetector.find_center(slice_hist, window, l_line)
            r_line = LaneDetector.find_center(slice_hist, window, r_line)
            line_centers.append((l_line, r_line))
        return line_centers

    @staticmethod
    def find_center(hist, window, center):
        window_width = len(window)
        window_width2 = window_width // 2
        line_min = max(center, 0)
        line_max = min(center + window_width, len(hist))
        if line_max - line_min >= window_width2 // 2:
            # TODO: need to handle off the screen lines
            center = line_min + np.argmax(np.convolve(hist[line_min:line_max], window)) - window_width2
        return center

    def highlight_features(self, img, line_centers):
        window_width2 = self.window_width // 2
        img_r, img_g, img_b = np.zeros_like(img), np.zeros_like(img), np.zeros_like(img)
        x_l, x_r, y = [], [], []

        for n, centers in enumerate(line_centers):
            bm = img.shape[0] - self.window_height * n
            tp = bm - self.window_height
            for center, img_ch in zip(centers, (img_r, img_b)):
                lt = max(center - window_width2, 0)
                rt = min(center + window_width2, img.shape[1])
                img_ch[tp:bm, lt:rt] = img[tp:bm, lt:rt] * 255
                if self.draw_convolution_windows:
                    cv2.rectangle(img_g, (lt, bm), (rt, tp), 255, thickness=2)
            x_l.append(centers[0])
            x_r.append(centers[1])
            v_center = tp + (bm - tp) // 2
            y.append(v_center)

        if not self.draw_convolution_windows:
            self.draw_lane(img_g, x_l, x_r, y)

        return np.dstack((img_r, img_g, img_b))

    @staticmethod
    def draw_lane(img, x_l, x_r, y):
        l_fit = np.polyfit(y, x_l, 2)
        r_fit = np.polyfit(y, x_r, 2)
        for y in range(0, img.shape[0]):
            y2 = y ** 2
            x_l = int(l_fit[0] * y2 + l_fit[1] * y + l_fit[2])
            x_r = int(r_fit[0] * y2 + r_fit[1] * y + r_fit[2])
            img[y, x_l:x_r] = 90

    def draw_hist(self, img, hist):
        hist = img.shape[0] - self.hist_height / np.max(hist) * hist
        cv2.rectangle(img, (0, img.shape[0] - self.hist_height), img.shape[1::-1], (0, 0, 0), cv2.FILLED)
        hist_fig = np.dstack((np.linspace(0, img.shape[1], img.shape[1]), hist))
        cv2.polylines(img, [hist_fig.astype(np.int32)], False, (255, 255, 255), thickness=2)
