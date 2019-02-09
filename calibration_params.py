import os
import pickle

CAMERA_MATRIX = 'camera_matrix'
DIST_COEFFS = 'dist_coeffs'

output_dir = 'calibration_output/'
file = output_dir + 'calibration.p'


def save(camera_matrix, dist_coeffs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pickle.dump({CAMERA_MATRIX: camera_matrix, DIST_COEFFS: dist_coeffs},
                open(file, 'wb'))


def load():
    if not os.path.exists(file):
        raise ValueError('File %s doesn\'t exist' % (file,))
    p = pickle.load(open(file, 'rb'))
    return p[CAMERA_MATRIX], p[DIST_COEFFS]
