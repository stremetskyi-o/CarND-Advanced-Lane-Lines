import glob
import os
import pickle
import sys

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def calibrate(files, pattern_sizes=((9, 6),)):
    print('Calibrating on %d images' % (len(files)))
    img_points = []
    obj_points = []
    img_size = None
    for file in files:
        img = mpimg.imread(file)
        cur_img_size = img.shape[1::-1]
        if img_size is None:
            img_size = cur_img_size
        elif img_size != cur_img_size:
            print('W: Different size in %s file, expected: %s, actual: %s' % (file, img_size, cur_img_size),
                  file=sys.stderr)
            continue
        for pattern_size in pattern_sizes:
            retval, corners = cv2.findChessboardCorners(img, pattern_size)
            if retval:
                obj_corners = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
                obj_corners[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
                img_points.append(corners)
                obj_points.append(obj_corners)
                break
            else:
                print('W: No corners found in %s file with pattern size %s' % (file, pattern_size), file=sys.stderr)

    retval, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    if retval:
        return camera_matrix, dist_coeffs
    else:
        raise ValueError('Unable to calibrate camera')


def main():
    files = glob.glob('camera_cal/calibration*.jpg')
    camera_matrix, dist_coeffs = calibrate(files, pattern_sizes=[(9, 6), (8, 6), (9, 5)])

    output_dir = 'calibration_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pickle.dump({'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs},
                open(output_dir + '/calibration.p', 'wb'))

    src = mpimg.imread(files[0])
    undistort = cv2.undistort(src, camera_matrix, dist_coeffs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(src)
    ax1.set_title('Original image')
    ax2.imshow(undistort)
    ax2.set_title('Undistorted image')
    plt.savefig('output_images/undistort_example.jpg', bbox_inches='tight')


if __name__ == "__main__":
    main()
