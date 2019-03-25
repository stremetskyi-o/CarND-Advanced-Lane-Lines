# **Advanced Lane Finding** 
## Project Writeup

---

The goals/steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistort1]: ./project-writeup-img/calibration1.jpg "Example of the distortion-corrected image"
[undistort2]: ./project-writeup-img/test2.jpg "Example of the distortion-corrected road image"
[colorspace]: ./project-writeup-img/good_test4.jpg "Good channels that display lane lines"
[gradient]: ./project-writeup-img/test2_2.jpg "Thresholded binary gradient image"
[perspective]: ./project-writeup-img/dst_straight_lines1.jpg "Perspective transform result"
[lane-lines-1]: ./project-writeup-img/test2_4.jpg "Binary image with lane lines detected and convolution windows drawn"
[lane-lines-2]: ./project-writeup-img/test2_5.jpg "Binary image with lane lines detected and line fits drawn"
[lane-lines-3]: ./project-writeup-img/test2_6.jpg "Binary image with lane lines detected and lane overlayed"
[result]: ./project-writeup-img/test2_7.jpg "Final result with lane detected and charachteristics annotated"
[challenge1]: ./project-writeup-img/1_challenge_video.mp4_snapshot_00.02_%5B2019.03.14_21.16.22%5D.jpg
 "Road image with dark lines"
[challenge2]: ./project-writeup-img/2_challenge_video.mp4_snapshot_00.02_%5B2019.03.14_21.16.22%5D.jpg
 "Binary gradient with negative values on the Red channel and positive on the Blue"
[challenge3]: ./project-writeup-img/3_challenge_video.mp4_snapshot_00.02_%5B2019.03.14_21.16.22%5D.jpg
 "Binary gradient with edges that lighter than surrounding"

---

### 1. Writeup

This writeup describes research and implementation steps that were taken to address project rubric points. Where 
appropriate steps are backed up by additional information in the form of images, tables, etc.

### 2. Camera Calibration

Camera calibration was performed on 20 images using OpenCV. Examining results for every image, it is seen that 3 images 
were rejected by the OpenCV algorithm. For this reason, the algorithm retries to find corners with a different pattern -
by decreasing the amount of horizontal or vertical corners by 1. This helped to pick-up corners in one more image but 
this particular algorithm rejected others. Also, some images have dimensions off by 1 pixel, it isn't much and 
shouldn't affect calibration, but higher orders could, so I consider it a good check for input images. The following is
the output of the described algorithm implemented in ```src/calibrate.py``` file:
```
Calibrating on 20 images
W: No corners found in camera_cal\calibration1.jpg file with pattern size (9, 6)
W: No corners found in camera_cal\calibration1.jpg file with pattern size (8, 6)
W: Different size in camera_cal\calibration15.jpg file, expected: (1280, 720), actual: (1281, 721)
W: No corners found in camera_cal\calibration4.jpg file with pattern size (9, 6)
W: No corners found in camera_cal\calibration4.jpg file with pattern size (8, 6)
W: No corners found in camera_cal\calibration4.jpg file with pattern size (9, 5)
W: No corners found in camera_cal\calibration5.jpg file with pattern size (9, 6)
W: No corners found in camera_cal\calibration5.jpg file with pattern size (8, 6)
W: No corners found in camera_cal\calibration5.jpg file with pattern size (9, 5)
W: Different size in camera_cal\calibration7.jpg file, expected: (1280, 720), actual: (1281, 721)
```
For every image where chessboards corners were found, I am storing image coordinates and preparing an XYZ array of 
corresponding object points, the X and Y coordinates are starting with 0 and incremented further, so every next column 
or row is one unit away from the previous. Z coordinate is always set to 0, meaning that all points are on the flat 
plane. Than image and object points are passed to `cv2.calibrateCamera()` function to calculate camera matrix and 
distortion coefficients.

For future use, the results of calibration are stored to a pickle file. The code responsible for storing and loading 
calibration parameters can be found in
`src/calibration_params.py` file.

Example of the distortion-corrected image using `cv2.undistort()`:

![undistort1]

### 3. Pipeline (single images)

#### 1. Distortion correction
The first thing to apply to input images is distortion correction; the values don't change between images from the same
camera, so it's safe to use values pre-computed and stored to the pickle file in the previous step. An example of 
distortion corrected road image:

![undistort2]

Before applying any modifications to the images I have started by coding up helper modules for calculating and
separating channels for RGB, HSL and HSV colorspaces, binary gradient with thresholds for X and Y, magnitude and
direction of the gradient, performing the perspective transform. The code for these operations can be found in 
`src/colorspace.py`, `src/gradient.py` and `src/perspective.py` respectively.

#### 2. Colorspaces and Gradient
I have started with testing which colorspaces and their channels constantly and distinctively display lane lines of
different color and under different light and shadow conditions. I have added a selection of additional images for
examination from the project videos to the number of images in `test_images` folder. For every image, I have
plotted separate channels of RGB, HSL and HSV colorspaces. Code for converting colorspaces and splitting channels is 
contained in `src/colorspace.py` file. After examination, I have split all colorspace/channel combinations to good
and bad ones, where Red channel of RGB, Saturation channel of HSL and V channel of HSV colorspaces are falling to the 
*good* category. An example of *good* combinations can be found on the image below.

![colorspace]

For the next steps I've selected HSL Saturation as it makes a good job getting lines under different conditions,
and RGB Red channel as it sometimes have more detail than Saturation on the far edge of the image. HSV Value was
discarded as it's mostly the same as R channel but requires additional colorspace transforming.

After preparing 1-bit images, it was possible to perform edge detection using the Sobel operator. I implemented the code
in `src/gradient.py` to calculate X and Y gradients, gradient magnitude and direction. Then performed thresholding to
make a binary gradient. I came up with the following scenario to choose the kernel size and thresholding parameters: 
iterate over range of the possible values for kernel size and first the low threshold and than high threshold and save
each frame to a GIF animation. Later I've examined animation to choose a preferable values using IrfanView so I can step
back and forth between frames as well as different files.

The kernel size was selected to remove the unevenness but to leave all the detail. Most images have good results with
kernel size 9 but direction of the gradient required bigger kernel size of 15. The low threshold was selected, so the
quality of the lines is not affected, yet most of the noise is removed; and the high threshold was set to the value
after which increasing the threshold doesn't give more detail.

The code for generating figures and animations as well as determining kernel size and thresholds can be found in
`evaluation_data.py`

Having taken a look at the results I had a feeling that most of the line features can be detected by X gradient alone, 
so I have combined X gradient of Red and Saturation channels by applying OR operator to them.

The result  can be seen in the next image:
![gradient]

#### 3. Perspective transform
I have implemented perspective transform in `src/perspective.py` file. The calculation is made with
`cv2.getPerspectiveTransform()`; the function takes a set of source and destination points and returns
the transformation matrix. Source points were selected in image editing software from the `straight_lines1.jpg` test
image and destination points derived from source points and shape of the input. The result is a bird's eye image of 
the road where straight lines appear to be parallel:

![perspective]

#### 4. Lane line detection
Detection of the lane lines is performed on the warped binary gradient image. The code for this step is in the
`find_lane_line_centers()` method of `LaneDetector` in `src/lane_detector.py`. First, the algorithm search for initial
line centers at the bottom of the image. To ensure good result, I am searching on the bottom third of the image by 
taking the sum of it on the horizontal axis and also multiplying the result by 3 and clipping it to original max value.
It is made to have similar peaks for solid and dashed lines. Then I am processing this histogram by convolving it with a
window of 1's using `np.convolve` and `scipy.signal.find_peaks` to get the reference center points. By passing 
*distance* parameter to `find_peaks` I am assuring that points lie at distance of at least minimal lane width. For the
points found I continue to run convolution with a smaller window around the previous point until nothing is found or top
of the image is reached. The result are estimated center points of the lane lines which cane be used to highlight actual
lane lines on the image. 

Center points are then passed to `highlight_lane_features` where a polynomial fit is calculated for the points, and lane
lines are highlighted on the image. The method can highlight features in different modes:
 
* Draw lane line windows for each center point:
![lane-lines-1]
* Draw fitted lines:
![lane-lines-2]
* Draw lane area
![lane-lines-3]

#### 5. Calculating feature characteristics
The characteristics being calculated are vehicle relative position to the lane and lane curvature. Both can be derived
from polynomial coefficients, but to receive the result in metric units, a conversion must be made. To get the
conversion ratio I used known features that have documented world size in X and Y directions, they are lane width and
length of the dashed line, by calculating size of the same features in pixels and dividing them, the desired ratios are
obtained.

Vehicle position is calculated from the center point of the lane and center of the image, assuming that camera is
mounted on the center of the vehicle(see `LaneDetector._lane_position()`. Line curvature is calculated in the point 
closest to the vehicle (see `LaneDetector._curve_radius()`. I am using modified equation, where A and B coefficients are
converted to the world space:
```python
a = fit.c[0] * self.m2p_ratio_x / self.m2p_ratio_y ** 2
b = fit.c[1] * self.m2p_ratio_x / self.m2p_ratio_y
r = (1 + (2 * a * y + b) ** 2) ** 1.5 / np.abs(2 * a)
```

#### 6. Overlay source image with detected features
Finally, I use `src/perspective.py`'s `unwarp` and `cv2.addWeighted` methods to get original image overlaid with the
highlighted features. The image is then annotated with information about vehicle position and line curvature using
`cv2.putText`. The final result is the following:

![result]

---

### Pipeline (video)

It is possible to apply the same techniques described in the previous section to a video stream, but for a better result
in terms of quality and speed, I've created a queue which holds data about 5 previous frames, which is enough for
the speed of the vehicle in the project videos. The data are: calculated centers of the lane lines and their fits. 
Previous line centers are used to estimate starting locations of the new ones in the method `_avg_line_center` of
`LaneDetector`. The history is checked if it is monotonic, if it is, then a linear extrapolation is applied to get the
new value, averaging is used otherwise. After calculating fit for the new frame, it gets averaged with the fits from the
history (in `_avg_line_fit` method of `LaneDetector`) making they less wobbly and more robust to the noise and false
detections.

Final video output can be found [here](./output_images/video/project_video.mp4).

---

### Discussion

The algorithm only covers some essential line detection and averaging along video sequence but successfully displays
which features can be derived from the video alone. Improved image processing together with some info about vehicle
speed and data from wheel angle sensor it is possible to receive a much better result. Some problems that algorithm may
face were detected from the testing on the challenge videos and some from the nature of the approach itself.

One of the moments where the algorithm can fail is where there are some clear dark lines parallel to the lane, like
on this image:
 
![challenge1]
 
It would be hard to diversify them if they have the same color and
shape. However, it is possible to diversify lighter and darker lines on the road using gradient. Here I've split 
gradient to the positive and negative parts and put them in different channels. Red for negative and blue for positive:

![challenge2]

Now it is seen that light edges have transitioned from positive to the negative side, and dark vise versa. So it is 
possible to tell which line is lighter and which is darker without taking any assumptions about color or color range.
The result of removing darker edges is the following:

![challenge3]

Another moment: is where the road makes sharp turns - algorithm doesn't always pick up such harsh changes but can be
adapted to do this. This also often causes one of the lines to go out of the camera view meaning that it must be derived
from previous frames. It can be displayed as the lane is going to the end of the image, but in fact it is not enough as
the line curvature may be wrong in this situation and it is required to extrapolate or average line fit. 

Some other drawbacks are connected to the speed of the vehicle, lane and lane lines characteristics, position of the
vehicle relative to the road, sun and weather conditions. Examples are:
* Dashed lines won't be successfully detected if the vehicle is moving too slowly
* Some complex lane lines may not be detected properly
* Lane lines won't be detected if the angle between them and vehicle is too big
* The algorithm is not tested at night, under rain and snow and will likely fail in this conditions. 

To overcome these issues it is required to have more footage for testing; I would also think about the following:
* which other features can help to stay in the lane if there is no lane lines or they are barely visible, like following
 the car ahead or detecting an edge of the road
* accurately detecting features that are part of more complex ones like road markings in form of polygons
* relying on other vehicle data: speed, wheel angle 

To improve the robustness of the algorithm a history size can be increased, yet the result will be less responsible to 
the fast road changes. Here it is possible to improve averaging by giving different weight's to the old and new result.