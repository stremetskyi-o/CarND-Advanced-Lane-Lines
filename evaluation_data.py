import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.fx.resize import resize as fx_resize
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import calibration_params
import colorspace
import gradient
from perspective import PerspectiveTransform


def evaluate_color_spaces(img, basename):
    # For testing purposes RGB, HLS and HSV color spaces were selected
    # Then they was plot on one figure and analyzed. I have also added
    # a sample image under the bridge from the challenge video.
    # Results:
    # * Hue component which is the same for the HLS and HSV has successful
    #   line detections but is too noisy.
    # * Blue channel doesn't pickup yellow lines
    # * Saturation channel of the HSV doesn't pickup white lines
    # * HSL Saturation channel makes a good job getting lines under different
    #   conditions, but sometimes loses detail on the far edge,
    # * Most detail can be also seen on R or V channels. R channel is preferable
    #   as there is no need to calculate HSV color space

    output_folder = 'output_images/colorspace/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    rgb_r, rgb_g, rgb_b = colorspace.rgb_channels(img)
    hsl_h, hsl_s, hsl_l = colorspace.hsl_channels(img)
    hsv_h, hsv_s, hsv_v = colorspace.hsv_channels(img)

    good_channels = [('Src', img), ('RGB: R', rgb_r), ('HSL: S', hsl_s), ('HSV: V', hsv_v)]
    bad_channels = [('Src', img), ('RGB: G', rgb_g), ('RGB: B', rgb_b), ('HSL: H', hsl_h),
                    ('HSL: L', hsl_l), ('HSV: S', hsv_s)]

    fig, subplots = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Good color spaces/channels: ' + basename)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for channel, subplot in zip(good_channels, subplots.flat):
        subplot.imshow(channel[1], cmap='gray')
        subplot.set_title(channel[0])

    plt.savefig(output_folder + 'good_%s' % (basename,))
    plt.close()

    fig, subplots = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('Bad color spaces/channels: ' + basename)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for channel, subplot in zip(bad_channels, subplots.flat):
        subplot.imshow(channel[1], cmap='gray')
        subplot.set_title(channel[0])

    plt.savefig(output_folder + 'bad_%s' % (basename,))
    plt.close()


def ensure_gradient_folder_exists():
    output_folder = 'output_images/gradient/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


def evaluate_gradient_kernel_size(img, basename):
    # I have started with coding functions for doing x,y gradient, magnitude and direction
    # Then started applying different kernel size and thresholds to get lines I am interested in.
    # First parameter for evaluation is kernel size. I have created gif animations with different values
    # to determine the best one.
    # After examining the animations I have got the following values:
    # 9 for x,y,mag, and 15 for direction

    output_folder = ensure_gradient_folder_exists()

    for channel, channel_name in zip((colorspace.rgb_channels(img)[0], colorspace.hsl_channels(img)[1]), ('R', 'S')):
        filename = '%s_%s' % (channel_name, basename)

        for grad_name, grad_func in (('1-X-K', gradient.abs_x),
                                     ('2-Y-K', gradient.abs_y),
                                     ('3-MAG-K', gradient.mag),
                                     ('4-DIR-K', lambda channel, ksize: gradient.g_dir(channel, ksize) * 80)):
            frames = []
            for ksize in range(3, 26, 2):
                frame = grad_func(channel, ksize)
                frame = np.dstack((frame, frame, frame))
                cv2.putText(frame, str(ksize), (1080, 100), cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), thickness=3)
                frames.append(frame)
            clip = ImageSequenceClip(frames, fps=3).fx(fx_resize, 0.5)
            clip.write_gif(output_folder + filename + '_%s.gif' % grad_name)
            clip.close()


def evaluate_gradient_variable_thresholds(img, fname, grad_func, ksize, threshold_variable, threshold_static,
                                          samples=None, side='l'):
    frames = []
    if not samples:
        samples = threshold_variable[1] - threshold_variable[0]

    for threshold_variable_step in np.linspace(threshold_variable[0], threshold_variable[1], samples):
        threshold = (threshold_variable_step, threshold_static)
        if side != 'l':
            threshold = threshold[::-1]
        frame = grad_func(img, ksize, threshold) * 255
        frame = np.dstack((frame, frame, frame))
        cv2.putText(frame, str(threshold_variable_step), (880, 100), cv2.FONT_HERSHEY_PLAIN, 8,
                    (255, 255, 255), thickness=3)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=3).fx(fx_resize, 0.5)
    clip.write_gif(fname)
    clip.close()


def evaluate_gradient_thresholds_l(img, basename):
    # Evaluation of the low threshold:
    # I decided to omit Y gradient as it hasn't been adding enough detail.
    # R channel:
    # * X - 40 , should be less (15-20) in difficult situation
    #   or even less under bridge (5)
    # * Magnitude - 80, should less for difficult situations as well (7 under bridge)
    # * Direction - 0.9
    #
    # S channel:
    # * Gradient X - can be a bit less (30), for light image a bit less, maybe (20-25)
    # * Gradient Magnitude also (30)
    # * Direction - 0.6-0.8
    #
    # S channel gets lines with same lower threshold of 25 well, except dark road where R channel with low value of 5-7
    # does better job

    output_folder = ensure_gradient_folder_exists()
    ksize = 9
    ksize_dir = 15

    rgb_r = colorspace.rgb_channels(img)[0]
    hsl_s = colorspace.hsl_channels(img)[1]

    evaluate_gradient_variable_thresholds(rgb_r, output_folder + 'R_%s_5-X-L.gif' % basename,
                                          gradient.abs_x, ksize, (4, 120), 255)
    evaluate_gradient_variable_thresholds(rgb_r, output_folder + 'R_%s_6-MAG-L.gif' % basename,
                                          gradient.mag, ksize, (4, 120), 255)
    evaluate_gradient_variable_thresholds(rgb_r, output_folder + 'R_%s_7-DIR-L.gif' % basename,
                                          gradient.g_dir, ksize_dir, (0, np.pi / 2), np.pi / 2, samples=20)
    evaluate_gradient_variable_thresholds(hsl_s, output_folder + 'S_%s_5-X-L.gif' % basename,
                                          gradient.abs_x, ksize, (4, 120), 255)
    evaluate_gradient_variable_thresholds(hsl_s, output_folder + 'S_%s_6-MAG-L.gif' % basename,
                                          gradient.mag, ksize, (4, 120), 255)
    evaluate_gradient_variable_thresholds(hsl_s, output_folder + 'S_%s_7-DIR-L.gif' % basename,
                                          gradient.g_dir, ksize_dir, (0, np.pi / 2), np.pi / 2, samples=20)


def evaluate_gradient_thresholds_h(img, basename):
    # Evaluation of the high threshold:
    # R channel:
    # * X, Magnitude - 170
    # * Direction - 1.2
    # S channel:
    # * X, Magnitude - 220
    # * Direction - 1.4

    output_folder = ensure_gradient_folder_exists()
    ksize = 9
    ksize_dir = 15

    rgb_r = colorspace.rgb_channels(img)[0]
    hsl_s = colorspace.hsl_channels(img)[1]

    evaluate_gradient_variable_thresholds(rgb_r, output_folder + 'R_%s_8-X-H.gif' % basename,
                                          gradient.abs_x, ksize, (50, 255), 5, side='h')
    evaluate_gradient_variable_thresholds(rgb_r, output_folder + 'R_%s_9-MAG-H.gif' % basename,
                                          gradient.mag, ksize, (50, 255), 7, side='h')
    evaluate_gradient_variable_thresholds(rgb_r, output_folder + 'R_%s_10-DIR-H.gif' % basename,
                                          gradient.g_dir, ksize_dir, (0.7, 1.3), 0.7, samples=15, side='h')
    evaluate_gradient_variable_thresholds(hsl_s, output_folder + 'S_%s_8-X-H.gif' % basename,
                                          gradient.abs_x, ksize, (100, 255), 25, side='h')
    evaluate_gradient_variable_thresholds(hsl_s, output_folder + 'S_%s_9-MAG-H.gif' % basename,
                                          gradient.mag, ksize, (100, 255), 25, side='h')
    evaluate_gradient_variable_thresholds(hsl_s, output_folder + 'S_%s_10-DIR-H.gif' % basename,
                                          gradient.g_dir, ksize_dir, (0.5, 1.5), 0.9, samples=15, side='h')


def evaluate_gradient_thresholds_lh(img, basename):
    # Generate figures with selected values for comparison
    # Red channel has 2 sets of thresholds - one for general conditions and another for low light

    output_folder = ensure_gradient_folder_exists()

    def generate_fig(channel, channel_name, ksize, ksize_dir, *thresholds):
        filename = '%s_%s' % (channel_name, basename)
        x = gradient.abs_x(channel, ksize, thresholds[0])
        mpimg.imsave(output_folder + filename, x, cmap='gray')
        mag = gradient.mag(channel, ksize, thresholds[1])
        g_dir = gradient.g_dir(channel, ksize_dir, thresholds[2])

        fig, subplots = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('%s gradients' % (channel_name,))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        subplots[0].imshow(x, cmap='gray')
        subplots[0].set_title('X')
        subplots[1].imshow(mag, cmap='gray')
        subplots[1].set_title('Magnitude')
        subplots[2].imshow(g_dir, cmap='gray')
        subplots[2].set_title('Direction')

        plt.savefig(output_folder + filename)
        plt.close()

    generate_fig(colorspace.rgb_channels(img)[0], 'R', 9, 15, (40, 170), (80, 170), (0.9, 1.2))
    generate_fig(colorspace.rgb_channels(img)[0], 'RL', 9, 15, (5, 170), (7, 170), (0.9, 1.2))
    generate_fig(colorspace.hsl_channels(img)[1], 'S', 9, 15, (25, 220), (25, 220), (0.7, 1.4))


def evaluate_perspective_transform(img, basename):
    output_folder = 'output_images/perspective/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    p = PerspectiveTransform()
    dst = p.warp(img)
    p.draw_dst(dst)
    p.draw_src(img)
    mpimg.imsave(output_folder + 'src_' + basename, img)
    mpimg.imsave(output_folder + 'dst_' + basename, dst)


if __name__ == '__main__':
    files = glob.glob('test_images/*.jpg')
    camera_matrix, dist_coeffs = calibration_params.load()
    for file in files:
        img = cv2.undistort(mpimg.imread(file), camera_matrix, dist_coeffs)
        basename = os.path.basename(file)
        evaluate_color_spaces(img, basename)
        evaluate_gradient_kernel_size(img, basename)
        evaluate_gradient_thresholds_l(img, basename)
        evaluate_gradient_thresholds_h(img, basename)
        evaluate_gradient_thresholds_lh(img, basename)
        evaluate_perspective_transform(img, basename)
