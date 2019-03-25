import glob
import os
import sys

import matplotlib.image as mpimg
from moviepy.video.io.VideoFileClip import VideoFileClip

from lane_detector import LaneDetector

if __name__ == '__main__':
    ld = LaneDetector()
    if len(sys.argv) > 1:
        output_folder = 'output_images/video/'
        for filename in sys.argv[1:]:
            clip = VideoFileClip(filename)
            out = clip.fl_image(ld.process_image)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            out.write_videofile(output_folder + filename, threads=4, audio=False)
            out.close()
    else:
        print('No file was specified. Creating example output...')
        files = glob.glob('test_images/*.jpg')
        for f in files:
            img = mpimg.imread(f)
            nformat = 'output_images/' + os.path.basename(f)[::-1].replace('.', '_%d.'[::-1], 1)[::-1]

            img = ld.undistort(img)
            mpimg.imsave(nformat % 1, img)

            lines_img = ld.binary_gradient(img)
            mpimg.imsave(nformat % 2, lines_img, cmap='gray')

            lines_img = ld.warp_perspective(lines_img)
            mpimg.imsave(nformat % 3, lines_img, cmap='gray')

            lane_lines = ld.find_lane_lines(lines_img)
            if lane_lines:
                feature_img = ld.highlight_lane_features(lines_img, lane_lines, mode='conv')
                mpimg.imsave(nformat % 4, feature_img)

                feature_img = ld.highlight_lane_features(lines_img, lane_lines, mode='fit')
                mpimg.imsave(nformat % 5, feature_img)

                feature_img = ld.highlight_lane_features(lines_img, lane_lines, mode='lane')
                mpimg.imsave(nformat % 6, feature_img)

                img = ld.unwarp_overlay(img, feature_img)
                ld.annotate(img, lane_lines)
                mpimg.imsave(nformat % 7, img)
