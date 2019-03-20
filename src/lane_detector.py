from collections import deque
from operator import itemgetter, attrgetter

import cv2
import numpy as np
from scipy import signal

import calibration_params
import colorspace
import gradient
from perspective import PerspectiveTransform

lane_width_m = 3.7
lane_width_p = 750
lane_dash_length_m = 3.0
lane_dash_length_p = 77
min_lane_width_p = 660


class FrameData:

    def __init__(self, line_centers, line_fits):
        self.line_centers = line_centers
        self.line_fits = line_fits


class LaneDetector:

    def __init__(self, window_width=100, window_height=80, history_size=5, draw_mode='lane', overlay_result=True):
        self.window_width = window_width
        self.window_width2 = window_width // 2
        self.window_width34 = window_width * 3 // 4
        self.window_height = window_height
        self.draw_mode = draw_mode
        self.overlay_result = overlay_result

        self.frames = deque(maxlen=history_size)
        self.calibration_params = calibration_params.load()
        self.perspective = PerspectiveTransform()

        self.m2p_ratio_x = lane_width_m / lane_width_p
        self.m2p_ratio_y = lane_dash_length_m / lane_dash_length_p

    def process_image(self, img):
        img = self.undistort(img)
        lines_img = self.binary_gradient(img)
        lines_img = self.warp_perspective(lines_img)

        line_centers = self.find_lane_line_centers(lines_img, history=self.frames)

        if line_centers is not None:
            lines_img, line_fits = self.highlight_lane_features(lines_img, line_centers, mode=self.draw_mode,
                                                                history=self.frames)
            self.frames.append(FrameData(line_centers, line_fits))

            if self.overlay_result:
                unwarped = self.unwarp_overlay(img, lines_img)
                self.annotate(unwarped, line_fits)
                return unwarped

            return lines_img
        else:
            if len(self.frames) > 0:
                self.frames.popleft()
            if self.overlay_result:
                return img
            else:
                return lines_img * 255

    def undistort(self, img):
        return cv2.undistort(img, *self.calibration_params)

    @staticmethod
    def binary_gradient(img):
        # Separate channels
        rgb_r = colorspace.rgb_channels(img)[0]
        hsl_s = colorspace.hsl_channels(img)[1]
        # Calculate and combine gradients
        r_x = gradient.abs_x(rgb_r, 9, (15, 170))
        s_x = gradient.abs_x(hsl_s, 9, (25, 220))
        lines_img = np.zeros_like(r_x)
        lines_img[(r_x == 1) | (s_x == 1)] = 1
        return lines_img

    def warp_perspective(self, lines_img):
        return self.perspective.warp(lines_img)

    def find_lane_line_centers(self, lines_img, history=None):
        window = np.ones(self.window_width)
        y = np.arange(lines_img.shape[0], -1, -self.window_height)

        line_centers_ref = self._calc_line_centers_ref(lines_img, window, y, history)
        if line_centers_ref is None:
            return None

        line_centers = [[line_centers_ref[i]] for i in range(len(line_centers_ref))]
        for i in range(1, len(y) - 1):
            bm = y[i]
            tp = y[i + 1]
            slice_hist = np.sum(lines_img[tp:bm], axis=0)
            for j in range(len(line_centers)):
                if len(line_centers[j]) == i:
                    center = self._find_next_center(slice_hist, window, line_centers[j][-1])
                    if center is not None:
                        line_centers[j].append(center)
        return line_centers

    def highlight_lane_features(self, img, line_centers, mode='lane', history=None):
        y = np.arange(img.shape[0], -1, -self.window_height)
        img_r, img_g, img_b = np.zeros_like(img), np.zeros_like(img), np.zeros_like(img)

        for i in range(len(y) - 1):
            bm = y[i]
            tp = y[i + 1]
            for centers, img_ch in zip(line_centers, (img_r, img_b)):
                if len(centers) > i:
                    lt = max(centers[i] - self.window_width2, 0)
                    rt = min(centers[i] + self.window_width2, img.shape[1])
                    img_ch[tp:bm, lt:rt] = img[tp:bm, lt:rt] * 255
                    if mode == 'conv':
                        cv2.rectangle(img_g, (lt, bm), (rt, tp), 255, thickness=2)

        line_fits = [np.poly1d(self._avg_line_fit(i, np.polyfit(y[:len(line_centers[i])], line_centers[i], 2), history))
                     for i in range(len(line_centers))]

        res = np.dstack((img_r, img_g, img_b))

        if mode == 'lane':
            self._draw_lane(res, line_fits)
        elif mode == 'fit':
            self._draw_lines(res, line_fits)

        return res, line_fits

    def unwarp_overlay(self, img, lines_img):
        lines_img = self.perspective.unwarp(lines_img)
        img = cv2.addWeighted(img, 0.8, lines_img, 1, 0)
        return img

    def annotate(self, img, line_fits):
        center_distance = self._lane_position(img, line_fits)
        if center_distance == 0:
            text1 = 'Vehicle is centered on the lane'
        else:
            side = 'left' if center_distance < 0 else 'right'
            text1 = 'Vehicle is %.2f m. %s of center' % (abs(center_distance), side)
        text2 = 'Curve radius is %.2f m.' % self._curve_radius(img.shape[0], line_fits)
        cv2.putText(img, text1, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (200, 200, 200),
                    thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, text2, (10, 110), cv2.FONT_HERSHEY_DUPLEX, 2, (200, 200, 200),
                    thickness=2, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_lane(img, line_fits):
        for y in range(0, img.shape[0]):
            x_l = int(line_fits[0](y))
            x_r = int(line_fits[1](y))
            img[y, x_l:x_r, 1] = 90

    @staticmethod
    def _draw_lines(img, line_fits):
        y = np.arange(img.shape[0])
        for fit in line_fits:
            x = fit(y)
            pts = np.column_stack((x, y)).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 0), thickness=3)

    def _calc_line_centers_ref(self, lines_img, window, y, history):
        if history is not None and len(history) > 0:
            hist = np.sum(lines_img[y[1]:y[0]], axis=0)
            line_centers_avg = [self._find_next_center(hist, window, self._avg_line_center(i, history))
                                for i in range(2)]
            if all(e is not None for e in line_centers_avg):
                return line_centers_avg

        # Calculate histogram to find start of the lanes at bottom of the image
        hist = np.sum(lines_img[lines_img.shape[0] * 2 // 3:], axis=0)
        # Multiplication and clipping allows to correctly detect dashed lines
        hist = np.clip(hist * 3, 0, np.max(hist))
        # Find initial left and right lines pixels using convolution
        convolve_signal = np.convolve(hist, window)
        peaks = signal.find_peaks(convolve_signal,
                                  distance=min_lane_width_p - self.window_width)[0] - self.window_width2
        if len(peaks) == 2:
            return peaks
        return None

    def _find_next_center(self, hist, window, center):
        line_min = max(center - self.window_width34, 0)
        line_max = min(center + self.window_width34, len(hist))
        if line_max - line_min >= self.window_width2 // 2:
            convolution_signal = np.convolve(hist[line_min:line_max], window, mode='same')
            if np.count_nonzero(convolution_signal) > 0:
                peaks = signal.find_peaks(convolution_signal)[0]
                if len(peaks) == 1:
                    return line_min + peaks[0]
        return None

    def _lane_position(self, img, line_fits):
        vehicle_x = (line_fits[0](img.shape[0]) + line_fits[1](img.shape[0])) / 2
        return (img.shape[1] / 2 - vehicle_x) * self.m2p_ratio_x

    def _curve_radius(self, y, line_fits):
        radius = []
        for fit in line_fits:
            a = fit.c[0] * self.m2p_ratio_x / self.m2p_ratio_y ** 2
            b = fit.c[1] * self.m2p_ratio_x / self.m2p_ratio_y
            r = (1 + (2 * a * y * self.m2p_ratio_y + b) ** 2) ** 1.5 / np.abs(2 * a)
            radius.append(r)
        return sum(radius) / len(radius)

    @staticmethod
    def monotonic(x):
        dx = np.diff(x)
        return np.all(dx <= 0) or np.all(dx >= 0)

    def _avg_line_center(self, idx, history):
        if history is None:
            return None
        old = list(map(itemgetter(0), map(itemgetter(idx), map(attrgetter('line_centers'), history))))
        if len(old) == 0:
            return None
        if len(old) > 1 and self.monotonic(old):
            linear_fit = np.poly1d(np.polyfit(np.arange(len(old)), old, 1))
            return int(linear_fit(len(old)))
        return sum(old) // len(old)

    @staticmethod
    def _avg_line_fit(idx, new, history):
        if history is None:
            return new
        old = list(map(attrgetter('c'), map(itemgetter(idx), map(attrgetter('line_fits'), history))))
        if len(old) == 0:
            return new
        return np.average(np.row_stack((old, new)), axis=0)
