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


class LaneLine:

    def __init__(self, points, line_fit):
        self.points = points
        self.line_fit = line_fit
        self.confident_fit = False


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

        lane_lines = self.find_lane_lines(lines_img, history=self.frames)

        if lane_lines is not None:
            lines_img = self.highlight_lane_features(lines_img, lane_lines, mode=self.draw_mode)
            self.frames.append(lane_lines)

            if self.overlay_result:
                unwarped = self.unwarp_overlay(img, lines_img)
                self.annotate(unwarped, lane_lines)
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

    def find_lane_lines(self, lines_img, history=None):
        window = np.ones(self.window_width)
        y = np.arange(lines_img.shape[0], -1, -self.window_height)

        left_pts = []
        right_pts = []
        line_pts = (left_pts, right_pts)

        if history and len(history):
            prev_fits = list(map(attrgetter('line_fit'), history[-1]))
            self._find_centers_range(lines_img, window, y, line_pts, True,
                                     lambda yi, li: int(prev_fits[li](y[yi])))

        if not len(left_pts) or not len(right_pts):
            line_centers_ref = self._calc_line_centers_ref(lines_img, window)
            if line_centers_ref is None:
                return None

            line_pts = [[(lines_img.shape[0], c)] for c in line_centers_ref]
            self._find_centers_range(lines_img, window, y[1:], line_pts, False,
                                     lambda yi, li: line_pts[li][-1][1])

        lane_lines = []
        for i in range(len(line_pts)):
            pts = np.array(line_pts[i])
            new_fit = np.polyfit(pts[:, 0], pts[:, 1], 2)
            avg_fit = np.poly1d(self._avg_line_fit(i, new_fit, history))
            lane_lines.append(LaneLine(pts, avg_fit))
        return tuple(lane_lines)

    def highlight_lane_features(self, img, lane_lines, mode='lane'):
        img_r, img_g, img_b = np.zeros_like(img), np.zeros_like(img), np.zeros_like(img)

        for i, lane_line, img_ch in zip(range(len(lane_lines)), lane_lines, (img_r, img_b)):
            pts = lane_line.points
            for pt in pts:
                bm = pt[0]
                tp = bm - self.window_height
                lt = max(pt[1] - self.window_width2, 0)
                rt = min(pt[1] + self.window_width2, img.shape[1])
                img_ch[tp:bm, lt:rt] = img[tp:bm, lt:rt] * 255
                if mode == 'conv':
                    cv2.rectangle(img_g, (lt, bm), (rt, tp), 255, thickness=2)

        res = np.dstack((img_r, img_g, img_b))

        if mode == 'lane':
            self._draw_lane(res, lane_lines)
        elif mode == 'fit':
            self._draw_lines(res, lane_lines)

        return res

    def unwarp_overlay(self, img, lines_img):
        lines_img = self.perspective.unwarp(lines_img)
        img = cv2.addWeighted(img, 0.8, lines_img, 1, 0)
        return img

    def annotate(self, img, lane_lines):
        line_fits = LaneDetector._map_line_fits(lane_lines)
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
    def _draw_lane(img, lane_lines):
        line_fits = LaneDetector._map_line_fits(lane_lines)
        for y in range(0, img.shape[0]):
            x_l = int(line_fits[0](y))
            x_r = int(line_fits[1](y))
            img[y, x_l:x_r, 1] = 90

    @staticmethod
    def _draw_lines(img, lane_lines):
        line_fits = LaneDetector._map_line_fits(lane_lines)
        y = np.arange(img.shape[0])
        for fit in line_fits:
            x = fit(y)
            pts = np.column_stack((x, y)).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 0), thickness=3)

    def _calc_line_centers_ref(self, lines_img, window):
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

    def _find_centers_range(self, lines_img, window, ys, line_pts, skip_allowed, prev_func):
        for yi in range(0, len(ys) - 1):
            bm = ys[yi]
            tp = ys[yi + 1]
            for li in range(len(line_pts)):
                pts = line_pts[li]
                if skip_allowed or len(pts) == yi:
                    center = None
                    prev_center = prev_func(yi, li)
                    line_min = max(prev_center - self.window_width34, 0)
                    line_max = min(prev_center + self.window_width34, lines_img.shape[1])
                    if line_max - line_min >= self.window_width2 // 2:
                        convolution_signal = np.convolve(np.sum(lines_img[tp:bm][:, line_min:line_max], axis=0), window,
                                                         mode='same')
                        if np.count_nonzero(convolution_signal) > 0:
                            peaks = signal.find_peaks(convolution_signal)[0]
                            if len(peaks) is not 0:
                                center = line_min + int(np.mean(peaks))
                    if center is not None:
                        pts.append((bm, center))

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

    @staticmethod
    def _avg_line_fit(idx, new, history):
        if history is None:
            return new
        old = list(map(attrgetter('c'), map(attrgetter('line_fit'), map(itemgetter(idx), history))))
        if len(old) == 0:
            return new
        return np.average(np.row_stack((old, new)), axis=0)

    @staticmethod
    def _map_line_fits(lane_lines):
        line_fits = list(map(attrgetter('line_fit'), lane_lines))
        return line_fits
