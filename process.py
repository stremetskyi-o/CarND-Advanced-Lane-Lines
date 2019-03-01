import os
import sys

from moviepy.video.io.VideoFileClip import VideoFileClip

from lane_detector import LaneDetector


if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_folder = 'output_images/video/'
        for filename in sys.argv[1:]:
            clip = VideoFileClip(filename)
            ld = LaneDetector()
            out = clip.fl_image(ld.process_image)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            out.write_videofile(output_folder + filename, threads=4, audio=False)
            out.close()
    else:
        print('No file was specified.')
