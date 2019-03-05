import cv2
import numpy as np
from scipy import signal

import calibration_params
import colorspace
import gradient
from perspective import PerspectiveTransform

lane_width_m = 3.7
lane_width_p = 740
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
        if line_centers is not None:
            lines_img, line_fits = self.find_features(lines_img, line_centers)

            if self.overlay_lanes:
                lines_img = self.perspective.unwarp(lines_img)
                img = cv2.addWeighted(img, 0.8, lines_img, 1, 0)
                self.draw_hist(img, hist)
                self.annotate_frame(img, line_fits)
                return img

            return lines_img
        else:
            if self.overlay_lanes:
                return img
            else:
                return lines_img * 255

    def find_centers(self, hist, lines_img, warped):
        window = np.ones(self.window_width)
        window_width2 = self.window_width // 2

        # Find initial left and right lines pixels using convolution
        convolve_signal = np.convolve(hist, window)
        peaks = signal.find_peaks(convolve_signal, distance=lane_width_p - self.window_width)[0] - window_width2
        if len(peaks) != 2:
            # TODO: Detect missing line(s) using previous frames
            return None
        l_line = peaks[0]
        r_line = peaks[1]
        # TODO: Average line centers using previous frames
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

    def find_features(self, img, line_centers):
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

        line_fits = (np.poly1d(np.polyfit(y, x_l, 2)),
                     np.poly1d(np.polyfit(y, x_r, 2)))

        if not self.draw_convolution_windows:
            self.draw_lane(img_g, line_fits)

        return np.dstack((img_r, img_g, img_b)), line_fits

    @staticmethod
    def draw_lane(img, line_fits):
        for y in range(0, img.shape[0]):
            x_l = int(line_fits[0](y))
            x_r = int(line_fits[1](y))
            img[y, x_l:x_r] = 90

    def draw_hist(self, img, hist):
        hist = img.shape[0] - self.hist_height / np.max(hist) * hist
        cv2.rectangle(img, (0, img.shape[0] - self.hist_height), img.shape[1::-1], (0, 0, 0), cv2.FILLED)
        hist_fig = np.dstack((np.linspace(0, img.shape[1], img.shape[1]), hist))
        cv2.polylines(img, [hist_fig.astype(np.int32)], False, (255, 255, 255), thickness=2)

    def annotate_frame(self, img, line_fits):
        vehicle_x = (line_fits[0](img.shape[0]) + line_fits[1](img.shape[0])) / 2
        center_distance = (img.shape[1] / 2 - vehicle_x) * self.m2p_ratio_x
        if center_distance == 0:
            text = 'Vehicle is centered on the lane'
        else:
            side = 'left' if center_distance < 0 else 'right'
            text = 'Vehicle is %.1f m. %s of center' % (abs(center_distance), side)
        cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (200, 200, 200), thickness=2, lineType=cv2.LINE_AA)
