
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Camera Calibration

The code for this step is contained in the first code cell of this IPython notebook.  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


```python
# Original chessboard image
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./camera_cal/calibration1.jpg')
print(image.shape)
plt.imshow(image)
```

    (720, 1280, 3)





    <matplotlib.image.AxesImage at 0x7f7d7b2a4d30>




![png](output_2_2.png)


**Some quick observations**

- Image size is consistent
- Number of intersections slightly differ. There would be the rare occurence where there is 9 by 5 x-y interactions on the chessboard. Normally, most of the images in the calibrartion folder is 9 by 6. The algorithm below is done such that it will ignore such cases


```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
%matplotlib inline

def get_chessboard_corners(fname, nx, ny):
    '''
    Convenience function for testing out chessboard detection
    Returns ret (boolean if result is available), corners coordinates and images
    '''
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    
    return ret, corners, img

def get_object_points(nx, ny):
    '''
    Function to get the object points needed to find the points on the chess board
    Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    '''
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    return objp
    
def get_obj_img_points(image_fname_list, nx, ny):
    '''
    Function to obtain the object and image point 
    '''
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Define a default object point
    objp = get_object_points(nx, ny) 
    
    # Step through the list and search for chessboard corners
    for fname in image_fname_list:
        # Find the chessboard corners
        ret, corners, mod_image = get_chessboard_corners(fname, nx, ny)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    return objpoints, imgpoints
            
```


```python
# Function testing - all are made to comments as they are only used for testing purposes

# Testing out the functions defined in the above cell
ret, corner, img = get_chessboard_corners('./camera_cal/calibration1.jpg', 9, 5)
```


```python
%matplotlib inline

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Get the object points and image points from a set of calibration images
objpoints, imgpoints = get_obj_img_points(images, 9, 6)

# Test undistortion on an image
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# Undistort
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Show image
plt.imshow(dst)

# Write the file out into the output folder
cv2.imwrite('./output_images/calibration1_undist.jpg',dst)
```




    True




![png](output_6_1.png)


## Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:





```python
# Original chessboard image
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Original Image
image = mpimg.imread('./test_images/test1.jpg')
print(image.shape)
plt.imshow(image)
```

    (720, 1280, 3)





    <matplotlib.image.AxesImage at 0x7f7d7b1f9940>




![png](output_8_2.png)



```python
# Undistorted Image
dst = cv2.undistort(image, mtx, dist, None, mtx)
plt.imshow(dst)
```




    <matplotlib.image.AxesImage at 0x7f7d7b165320>




![png](output_9_1.png)


**Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.**

I used a combination of color and gradient thresholds to generate a binary image. In order to test the combinations of binary images at a faster pace, we would create all the required thresholding functions which would include the following:

- rgb threshold functions. They deal with trying to get the image into a single color channel and see if using such a color channel would be able to provide enough information in order to get the lane lines
- hls threshold functions. They deal with trying to get the image into a single color channel but instead of the usual rgb channel, they deal with the hls channel. They will see if using such a color channel would be able to provide enough information in order to get the lane lines
- direction threshold function. They deal with changes in the direction of gradient change of the image
- magniture threshold function. They deal with changes in the magnitude of gradient change of the image
- sobel x threshold. Utiilizes sobel algorithm to get the gradients in the image by the x axis of the image
- sobel y threshold. Utiilizes sobel algorithm to get the gradients in the image by the y axis of the image


```python
# Define color threshold functions - Can choose to utilize which one of the color gradients work best

def rgb_threshold(image, filter=None, mode=None, lower_threshold=None, upper_threshold=None):
    '''
    Provide RGB image for processing. Allows one to select color filter and even color thresholding
    '''
    # Apply color filter
    if filter == 'r':
        image = image[:,:,0]
    if filter == 'g':
        image = image[:,:,1]
    if filter == 'b':
        image = image[:,:,2]
    
    # Allow user to select to binarize the image based on a threshold
    if mode == 'binary':
        binary_image = np.zeros_like(image)
        binary_image[(image > lower_threshold) & (image <= upper_threshold)] = 1
        return binary_image
    else:
        return image

def hls_threshold(image, filter=None, mode=None, lower_threshold=None, upper_threshold=None):
    '''
    Provide RGB image for processing. Image will be converted to hls image. Allow one to select hls filter and hls
    thresholding
    '''
    # Convert to hls image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # Apply color filter
    if filter == 'h':
        image = image[:,:,0]
    if filter == 'l':
        image = image[:,:,1]
    if filter == 's':
        image = image[:,:,2]
    
    # Allow user to select to binarize the image based on a threshold
    if mode == 'binary':
        binary_image = np.zeros_like(image)
        binary_image[(image > lower_threshold) & (image <= upper_threshold)] = 1
        return binary_image
    else:
        return image
```


```python
road_image = mpimg.imread("./test_images/straight_lines1.jpg")
road_image = cv2.undistort(road_image, mtx, dist, None, mtx)
```


```python
# Testing with just red filter doesn't work
# Test on every single color reveals that none of them work really well because of the shadow. 
# Red may work with some images but it can miserably fail at some of them
rgb_binary = rgb_threshold(road_image, 'r', 'binary', 60, 200)
plt.imshow(rgb_binary, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f7d7b0d2320>




![png](output_13_1.png)



```python
# Testing with just red filter doesn't work
#road_image = mpimg.imread("./test_images/test5.jpg")
hls_binary = hls_threshold(road_image, 's', 'binary', 50, 220)
plt.imshow(hls_binary, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f7d7b0bf080>




![png](output_14_1.png)



```python
# Define a function called sobel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output
```


```python
#road_image = mpimg.imread("./test_images/test1.jpg")
gradx = abs_sobel_thresh(road_image, 'x', 5, 200)
plt.imshow(gradx, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f7d7b024d68>




![png](output_16_1.png)



```python
#road_image = mpimg.imread("./test_images/test5.jpg")
grady = abs_sobel_thresh(road_image, 'y', 10, 200)
plt.imshow(grady, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f7d7af8eac8>




![png](output_17_1.png)



```python
#road_image = mpimg.imread("./test_images/test1.jpg")
dir_binary = dir_threshold(road_image, 3, (0.6, 0.7))
plt.imshow(dir_binary, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f7d7af7a748>




![png](output_18_1.png)



```python
#road_image = mpimg.imread("./test_images/test5.jpg")
mag_binary = mag_thresh(road_image,3, mag_thresh=(15,30))
plt.imshow(mag_binary, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f7d7aee64e0>




![png](output_19_1.png)



```python
combined = np.zeros_like(rgb_binary)
# combined[((gradx == 1) & (grady == 1)) | 
#        ((mag_binary == 1) & (dir_binary == 0)) | 
#          ((rgb_binary == 0) & (hls_binary == 1))] = 1
# combined[((rgb_binary == 1) | (hls_binary == 1))] = 1
combined[((rgb_binary == 0) & (gradx == 1)) |
         ((rgb_binary == 0) & (grady == 1))] = 1
plt.imshow(combined, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f7d7ae50518>




![png](output_20_1.png)


**Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.**

The code for my perspective transform includes a function called `perpective_warper()`, which appears in the next codeblock.  The `perspective_warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:


```python
road_image = mpimg.imread("./test_images/straight_lines1.jpg")
# Undistort
dst = cv2.undistort(road_image, mtx, dist, None, mtx)
img_size = dst.shape

# # Zoomed-out version
# src = np.float32(
# [[580, 450],
# [40, 720],
# [1240, 720],
# [700, 450]])

# # Zoomed-in version
# src = np.float32(
# [[600, 450],
# [200, 720],
# [1000, 720],
# [680, 450]])

# Zoomed-in version
src = np.float32(
[[520, 500],
[200, 720],
[1000, 720],
[730, 500]])

# dst = np.float32(
#     [[0, 0],
#     [0, 720],
#     [1280, 720],
#     [1280, 0]])

dst = np.float32(
    [[(img_size[1] / 4), 0],
    [(img_size[1] / 4), img_size[0]],
    [(img_size[1] * 3 / 4), img_size[0]],
    [(img_size[1] * 3 / 4), 0]])

# For use in the later portion of this notebook
Minv = cv2.getPerspectiveTransform(dst, src)

def perspective_wrapper(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    perspective_transform = cv2.warpPerspective(img, M, img_size)
    
    return perspective_transform
```


```python
warped_image = perspective_wrapper(road_image, src, dst)
plt.imshow(warped_image)
```




    <matplotlib.image.AxesImage at 0x7f7d7ae3c470>




![png](output_23_1.png)


**Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?**

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:


```python
# Function to get the starting point of the function
def get_start_points(binary_warped):
    '''
    Return the starting point for the left and right lanes
    '''
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base

def get_initial_indices(binary_warped, leftx_base, rightx_base, nwindows=9, margin = 100, minpix = 50):
    '''
    Get initial left and right lane indices
    '''
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return out_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy

def plot_lane_fit(binary_warped, out_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    '''
    Plot results form previous lane line calculations
    '''
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Color left lane points red
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # Color right lane points blue
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return out_img
    
def get_indices(binary_warped, left_fit, right_fit, margin=100):
    '''
    Get the left and right indices 
    '''
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting

    return left_fit, right_fit, left_fitx, right_fitx, nonzerox, nonzeroy, left_lane_inds, right_lane_inds
```


```python
%matplotlib inline

# Get the warped image from above
binary_warped = perspective_wrapper(combined, src, dst)
plt.imshow(binary_warped, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f7d7ada7080>




![png](output_26_1.png)



```python
%matplotlib inline

# Run the above functions to get the polyfits for the following road
leftx_base, rightx_base = get_start_points(binary_warped)
out_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = process_image(binary_warped, leftx_base, rightx_base)
out_img = plot_lane_fit(binary_warped, out_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
```

    /root/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/ipykernel/__main__.py:6: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future





    (720, 0)




![png](output_27_2.png)



```python
left_fit, right_fit, left_fitx, right_fitx, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = get_indices(binary_warped, left_fit, right_fit)

# Create an image to draw on and an image to show the selection window
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
```




    (720, 0)




![png](output_28_1.png)


**Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.**

I did this in the next following block of code


```python
def get_pixel_curvature(max_height, left_fit, right_fit):
    '''
    Generate the curvature based on pixel dimensions
    '''
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad, right_curverad
    
def get_world_curvature(nonzerox, nonzeroy, left_lane_inds, right_lane_inds, ym_per_pix, xm_per_pix):
    y_eval = 719
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad
```


```python
# Test out functions defined above
get_pixel_curvature(719, left_fit, right_fit)
```




    (91860.168474785081, 20328.737826094646)




```python
xm_per_pix = 3.7/500
ym_per_pix = 3/100
get_world_curvature(nonzerox, nonzeroy, left_lane_inds, right_lane_inds, ym_per_pix, xm_per_pix)
```




    (11164.622446955329, 2472.1059887514693)



**Observations**

Although, world curvature looks parallel at first glance, when calculations are done, it shows huge differences in the world curvature. This is kind of expected, seeing that this stretch of road is quite straight. Any slight deviations would lead to huge diffences in curvature.

**Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.**

I implemented this step in next set of code blocks. You can view the image for the lane there.


```python
# Function to generate the x points when given a fit object
def generate_fitx_values(fit):
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    return fitx
```


```python
# Initialization to draw the image on
warped = binary_warped
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Get the x-y coordinates on where to draw the lines
ploty = np.linspace(0, 719, num=720)
left_fitx = generate_fitx_values(left_fit)
right_fitx = generate_fitx_values(right_fit)

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (road_image.shape[1], road_image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(road_image, 1, newwarp, 0.3, 0)
plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x7f7d7ac79e48>




![png](output_36_1.png)



```python
# Define a class to receive the characteristics of each line detection
# This class will only be limited to get information, and not to do calculation
# To store the details of the left line and right line separately
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # Store last n set of fits
        self.fits = []
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # value of n
        self.n = 5
        self.realWorldYMul = 3/100 # Denomintator is no of pixels covered. For Y is the length of a lane
        self.realWorldXMul = 3.7/500 # Denominator is no of pixels covered. For X is the width of a standard lane
        
    def lineDetected(self, detected):
        '''
        Store boolean on whether line was detected or not
        '''
        self.detected = detected
        
    def storeFits(self, fit):
        '''
        Store last n iterations of fits
        '''
        self.fits.append(fit)
        if len(self.fits) >=  self.n:
            self.fits.pop(0)
        
    def storeXFitted(self, fit):
        '''
        Add new value to the list at the back. If there were more than n fits of the line, remove first instance
        This is not getting the fit but rather the x values produced by the fit
        '''
        no_of_fits = len(self.fits)
        ploty = np.linspace(0, len(self.fits[(no_of_fits - 1)]) - 1, num=len(self.fits[(no_of_fits - 1)]))
        xfitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        self.recent_xfitted.append(xfitx)
        if len(self.recent_xfitted) >= self.n:
            self.recent_xfitted.pop(0)
        
    def getBestX(self):
        '''
        Get the best set of x values of the lines
        '''
        self.bestx = np.mean((self.recent_xfitted), 0)
        
    def getBestFit(self):
        '''
        Get the best fit coordiantes by using the best x coordinates
        '''
        ploty = np.linspace(0, len(self.bestx) - 1, num=len(self.bestx))
        fit = np.polyfit(ploty, self.bestx, 2)
        self.current_fit = fit
        
    def getRadiusOfCurvature(self):
        '''
        Get the real world radius of curvature
        '''
        y_points = np.linspace(0, len(self.bestx) - 1, num=len(self.bestx))
        x_points = self.bestx
        y_points = y_points * self.realWorldYMul
        x_points = x_points * self.realWorldXMul
        fit_cr = np.polyfit(lefty, leftx, 2)
        y_eval = max(y_points)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*self.realWorldYMul + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        
    def getBaseLinePos(self, xfitx_secondary):
        '''
        Parameter to accept the second line
        '''
        related_coordinate = len(self.bestx) - 1
        base_point_second_line = xfitx_secondary[related_coordinate]
        base_point = self.bestx[related_coordinate]
        distance_in_pixels = abs(base_point - base_point_second_line)
        self.line_base_pos = distance_in_pixels * self.realWorldXMul
        
    def getDiffs(self):
        current_no = len(self.fits)
        if current_no >= 2:
            self.diffs = self.fits[current_no - 1] - self.fits[current_no - 2]
            
    def setAllX(self, x_pixels):
        self.allx = x_pixels
        
    def setAllY(self, y_pixels):
        self.ally = y_pixels
        
    
```

## Pipeline (video)

Here's a [link to my video result](./project_video.mp4)
TODO: Need to add the correct video link


```python
# Code to generate video
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
# Declaring additional functions that were not declared or declared improperly above
def thresholding(image):
    '''
    Comment/Uncomment sections that are not really required
    '''
    rgb_binary = rgb_threshold(road_image, 'r', 'binary', 60, 200)
    hls_binary = hls_threshold(road_image, 's', 'binary', 50, 220)
    gradx = abs_sobel_thresh(road_image, 'x', 5, 200)
    grady = abs_sobel_thresh(road_image, 'y', 10, 200)
    dir_binary = dir_threshold(road_image, 3, (0.6, 0.7))
    mag_binary = mag_thresh(road_image,3, mag_thresh=(15,30))
    
    combined = np.zeros_like(rgb_binary)
    combined[((rgb_binary == 0) & (gradx == 1)) |
             ((rgb_binary == 0) & (grady == 1))] = 1
    
    return combined
```


```python
# Declaring additional functions that were not declared or declared improperly above
def remapLaneLine(original_image, binary_warped, left_fit, right_fit):
    # Initialization to draw the image on
    warped = binary_warped
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Get the x-y coordinates on where to draw the lines
    ploty = np.linspace(0, 719, num=720)
    left_fitx = generate_fitx_values(left_fit)
    right_fitx = generate_fitx_values(right_fit)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return result
```


```python
# Declare left line and right line
left_line = Line()
right_line = Line()
```


```python
# Some of the variables have to be obtained from the environment
def process_image(image):
    '''
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    The following function will replicate every step taken along the way to reach the end of overlaying the
    lane lines on the video
    '''
    # Distortion Correction
    image_undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    
    # Color/Gradient Thresholding
    image_thresholded = thresholding(image_undistorted)
    
    # Perspective Transform
    image_warped = perspective_wrapper(image_thresholded, src, dst)
    
    # Reading the data from the class that records the lines
    # If either line was not detected, need to recalibrate the initial points
    if (not left_line.detected) | (not right_line.detected):
        leftx_base, rightx_base = get_start_points(image_warped)
        out_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = get_initial_indices(image_warped, leftx_base, rightx_base, nwindows=9, margin = 100, minpix = 50)
    else:
        try:
            left_fit = left_line.best_fit
            right_fit = right_line.best_fit
            left_fit, right_fit, left_fitx, right_fitx, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = get_indices(image_warped, left_fit, right_fit, margin=100)
        except:
            leftx_base, rightx_base = get_start_points(image_warped)
            out_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = get_initial_indices(image_warped, leftx_base, rightx_base, nwindows=9, margin = 100, minpix = 50)

    # Create the final color image
    finalImage = remapLaneLine(image_undistorted, image_warped, left_fit, right_fit)
    
    # Record all left lane info
    left_line.lineDetected(True)
    left_line.storeFits(left_fit)
    left_line.storeXFitted(left_fit)
    left_line.getBestX()
    left_line.getBestFit()
    left_line.getRadiusOfCurvature()
    left_line.getDiffs()
    
    # Record all right lane info
    right_line.lineDetected(True)
    right_line.storeFits(left_fit)
    right_line.storeXFitted(left_fit)
    right_line.getBestX()
    right_line.getBestFit()
    right_line.getRadiusOfCurvature()
    right_line.getDiffs()
    
    # Record the base line position
    left_line.getBaseLinePos(right_line.bestx)
    right_line.getBaseLinePos(left_line.bestx)
    
    return finalImage
```


```python
video_output = 'lane_lines_output.mp4'
clip1 = VideoFileClip("project_video.mp4") # Input
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video lane_lines_output.mp4
    [MoviePy] Writing video lane_lines_output.mp4


    
    
      0%|          | 0/1261 [00:00<?, ?it/s][A[A
    
      0%|          | 1/1261 [00:00<04:23,  4.78it/s][A[A
    
      0%|          | 2/1261 [00:00<04:37,  4.53it/s][A[A
    
      0%|          | 3/1261 [00:00<04:28,  4.68it/s][A[A
    
      0%|          | 4/1261 [00:00<04:24,  4.75it/s][A[A
    
      0%|          | 5/1261 [00:01<04:15,  4.92it/s][A[A
    
      0%|          | 6/1261 [00:01<04:10,  5.01it/s][A[A
    
      1%|          | 7/1261 [00:01<04:08,  5.04it/s][A[A
    
      1%|          | 8/1261 [00:01<04:03,  5.14it/s][A[A
    
      1%|          | 9/1261 [00:01<04:01,  5.18it/s][A[A
    
      1%|          | 10/1261 [00:02<04:10,  5.00it/s][A[A
    
      1%|          | 11/1261 [00:02<04:04,  5.11it/s][A[A
    
      1%|          | 12/1261 [00:02<04:02,  5.15it/s][A[A
    
      1%|          | 13/1261 [00:02<03:58,  5.24it/s][A[A
    
      1%|          | 14/1261 [00:02<03:55,  5.30it/s][A[A
    
      1%|          | 15/1261 [00:02<03:59,  5.21it/s][A[A
    
      1%|▏         | 16/1261 [00:03<03:56,  5.26it/s][A[A
    
      1%|▏         | 17/1261 [00:03<03:56,  5.26it/s][A[A
    
      1%|▏         | 18/1261 [00:03<03:56,  5.25it/s][A[A
    
      2%|▏         | 19/1261 [00:03<03:55,  5.27it/s][A[A
    
      2%|▏         | 20/1261 [00:03<03:53,  5.32it/s][A[A
    
      2%|▏         | 21/1261 [00:04<03:52,  5.34it/s][A[A
    
      2%|▏         | 22/1261 [00:04<03:51,  5.35it/s][A[A
    
      2%|▏         | 23/1261 [00:04<03:52,  5.32it/s][A[A
    
      2%|▏         | 24/1261 [00:04<03:52,  5.32it/s][A[A
    
      2%|▏         | 25/1261 [00:04<03:51,  5.33it/s][A[A
    
      2%|▏         | 26/1261 [00:05<03:55,  5.24it/s][A[A
    
      2%|▏         | 27/1261 [00:05<04:00,  5.12it/s][A[A
    
      2%|▏         | 28/1261 [00:05<04:05,  5.02it/s][A[A
    
      2%|▏         | 29/1261 [00:05<04:12,  4.89it/s][A[A
    
      2%|▏         | 30/1261 [00:05<04:10,  4.92it/s][A[A
    
      2%|▏         | 31/1261 [00:06<04:13,  4.86it/s][A[A
    
      3%|▎         | 32/1261 [00:06<04:14,  4.83it/s][A[A
    
      3%|▎         | 33/1261 [00:06<04:16,  4.78it/s][A[A
    
      3%|▎         | 34/1261 [00:06<04:17,  4.77it/s][A[A
    
      3%|▎         | 35/1261 [00:06<04:15,  4.80it/s][A[A
    
      3%|▎         | 36/1261 [00:07<04:13,  4.84it/s][A[A
    
      3%|▎         | 37/1261 [00:07<04:07,  4.94it/s][A[A
    
      3%|▎         | 38/1261 [00:07<04:00,  5.08it/s][A[A
    
      3%|▎         | 39/1261 [00:07<03:56,  5.17it/s][A[A
    
      3%|▎         | 40/1261 [00:07<03:54,  5.21it/s][A[A
    
      3%|▎         | 41/1261 [00:08<03:56,  5.16it/s][A[A
    
      3%|▎         | 42/1261 [00:08<03:53,  5.23it/s][A[A
    
      3%|▎         | 43/1261 [00:08<03:51,  5.25it/s][A[A
    
      3%|▎         | 44/1261 [00:08<03:49,  5.29it/s][A[A
    
      4%|▎         | 45/1261 [00:08<03:48,  5.31it/s][A[A
    
      4%|▎         | 46/1261 [00:09<03:47,  5.35it/s][A[A
    
      4%|▎         | 47/1261 [00:09<03:45,  5.39it/s][A[A
    
      4%|▍         | 48/1261 [00:09<03:44,  5.41it/s][A[A
    
      4%|▍         | 49/1261 [00:09<03:44,  5.41it/s][A[A
    
      4%|▍         | 50/1261 [00:09<03:45,  5.38it/s][A[A
    
      4%|▍         | 51/1261 [00:09<03:51,  5.23it/s][A[A
    
      4%|▍         | 52/1261 [00:10<03:47,  5.30it/s][A[A
    
      4%|▍         | 53/1261 [00:10<03:50,  5.24it/s][A[A
    
      4%|▍         | 54/1261 [00:10<03:51,  5.21it/s][A[A
    
      4%|▍         | 55/1261 [00:10<03:48,  5.27it/s][A[A
    
      4%|▍         | 56/1261 [00:10<03:48,  5.28it/s][A[A
    
      5%|▍         | 57/1261 [00:11<03:49,  5.25it/s][A[A
    
      5%|▍         | 58/1261 [00:11<03:46,  5.31it/s][A[A
    
      5%|▍         | 59/1261 [00:11<03:47,  5.29it/s][A[A
    
      5%|▍         | 60/1261 [00:11<03:45,  5.32it/s][A[A
    
      5%|▍         | 61/1261 [00:11<03:44,  5.34it/s][A[A
    
      5%|▍         | 62/1261 [00:12<03:47,  5.28it/s][A[A
    
      5%|▍         | 63/1261 [00:12<03:55,  5.09it/s][A[A
    
      5%|▌         | 64/1261 [00:12<03:59,  4.99it/s][A[A
    
      5%|▌         | 65/1261 [00:12<03:57,  5.04it/s][A[A
    
      5%|▌         | 66/1261 [00:12<03:59,  4.98it/s][A[A
    
      5%|▌         | 67/1261 [00:13<04:02,  4.93it/s][A[A
    
      5%|▌         | 68/1261 [00:13<04:19,  4.60it/s][A[A
    
      5%|▌         | 69/1261 [00:13<04:08,  4.80it/s][A[A
    
      6%|▌         | 70/1261 [00:13<04:02,  4.90it/s][A[A
    
      6%|▌         | 71/1261 [00:13<04:03,  4.89it/s][A[A
    
      6%|▌         | 72/1261 [00:14<03:56,  5.04it/s][A[A
    
      6%|▌         | 73/1261 [00:14<03:57,  5.00it/s][A[A
    
      6%|▌         | 74/1261 [00:14<03:59,  4.97it/s][A[A
    
      6%|▌         | 75/1261 [00:14<03:58,  4.98it/s][A[A
    
      6%|▌         | 76/1261 [00:14<03:58,  4.98it/s][A[A
    
      6%|▌         | 77/1261 [00:15<04:01,  4.91it/s][A[A
    
      6%|▌         | 78/1261 [00:15<03:59,  4.94it/s][A[A
    
      6%|▋         | 79/1261 [00:15<03:57,  4.97it/s][A[A
    
      6%|▋         | 80/1261 [00:15<04:00,  4.91it/s][A[A
    
      6%|▋         | 81/1261 [00:15<04:06,  4.78it/s][A[A
    
      7%|▋         | 82/1261 [00:16<03:59,  4.93it/s][A[A
    
      7%|▋         | 83/1261 [00:16<04:01,  4.88it/s][A[A
    
      7%|▋         | 84/1261 [00:16<04:06,  4.78it/s][A[A
    
      7%|▋         | 85/1261 [00:16<04:07,  4.75it/s][A[A
    
      7%|▋         | 86/1261 [00:16<04:11,  4.68it/s][A[A
    
      7%|▋         | 87/1261 [00:17<04:05,  4.77it/s][A[A
    
      7%|▋         | 88/1261 [00:17<04:04,  4.79it/s][A[A
    
      7%|▋         | 89/1261 [00:17<04:03,  4.81it/s][A[A
    
      7%|▋         | 90/1261 [00:17<03:59,  4.88it/s][A[A
    
      7%|▋         | 91/1261 [00:17<03:59,  4.90it/s][A[A
    
      7%|▋         | 92/1261 [00:18<03:55,  4.96it/s][A[A
    
      7%|▋         | 93/1261 [00:18<03:59,  4.88it/s][A[A
    
      7%|▋         | 94/1261 [00:18<04:01,  4.84it/s][A[A
    
      8%|▊         | 95/1261 [00:18<04:02,  4.82it/s][A[A
    
      8%|▊         | 96/1261 [00:19<03:58,  4.88it/s][A[A
    
      8%|▊         | 97/1261 [00:19<03:56,  4.91it/s][A[A
    
      8%|▊         | 98/1261 [00:19<03:50,  5.05it/s][A[A
    
      8%|▊         | 99/1261 [00:19<03:45,  5.14it/s][A[A
    
      8%|▊         | 100/1261 [00:19<03:43,  5.21it/s][A[A
    
      8%|▊         | 101/1261 [00:19<03:43,  5.19it/s][A[A
    
      8%|▊         | 102/1261 [00:20<03:42,  5.22it/s][A[A
    
      8%|▊         | 103/1261 [00:20<03:39,  5.26it/s][A[A
    
      8%|▊         | 104/1261 [00:20<03:43,  5.19it/s][A[A
    
      8%|▊         | 105/1261 [00:20<03:44,  5.15it/s][A[A
    
      8%|▊         | 106/1261 [00:20<03:42,  5.19it/s][A[A
    
      8%|▊         | 107/1261 [00:21<03:40,  5.24it/s][A[A
    
      9%|▊         | 108/1261 [00:21<03:47,  5.06it/s][A[A
    
      9%|▊         | 109/1261 [00:21<03:52,  4.95it/s][A[A
    
      9%|▊         | 110/1261 [00:21<03:51,  4.97it/s][A[A
    
      9%|▉         | 111/1261 [00:21<04:00,  4.78it/s][A[A
    
      9%|▉         | 112/1261 [00:22<04:02,  4.74it/s][A[A
    
      9%|▉         | 113/1261 [00:22<04:06,  4.66it/s][A[A
    
      9%|▉         | 114/1261 [00:22<04:02,  4.73it/s][A[A
    
      9%|▉         | 115/1261 [00:22<04:03,  4.71it/s][A[A
    
      9%|▉         | 116/1261 [00:23<04:00,  4.77it/s][A[A
    
      9%|▉         | 117/1261 [00:23<04:02,  4.73it/s][A[A
    
      9%|▉         | 118/1261 [00:23<04:10,  4.56it/s][A[A
    
      9%|▉         | 119/1261 [00:23<04:08,  4.60it/s][A[A
    
     10%|▉         | 120/1261 [00:23<04:07,  4.62it/s][A[A
    
     10%|▉         | 121/1261 [00:24<04:05,  4.64it/s][A[A
    
     10%|▉         | 122/1261 [00:24<03:58,  4.77it/s][A[A
    
     10%|▉         | 123/1261 [00:24<03:57,  4.78it/s][A[A
    
     10%|▉         | 124/1261 [00:24<04:00,  4.73it/s][A[A
    
     10%|▉         | 125/1261 [00:24<03:51,  4.91it/s][A[A
    
     10%|▉         | 126/1261 [00:25<03:49,  4.95it/s][A[A
    
     10%|█         | 127/1261 [00:25<03:53,  4.86it/s][A[A
    
     10%|█         | 128/1261 [00:25<03:50,  4.92it/s][A[A
    
     10%|█         | 129/1261 [00:25<03:54,  4.83it/s][A[A
    
     10%|█         | 130/1261 [00:25<03:48,  4.95it/s][A[A
    
     10%|█         | 131/1261 [00:26<03:44,  5.03it/s][A[A
    
     10%|█         | 132/1261 [00:26<03:38,  5.16it/s][A[A
    
     11%|█         | 133/1261 [00:26<03:45,  5.01it/s][A[A
    
     11%|█         | 134/1261 [00:26<03:49,  4.90it/s][A[A
    
     11%|█         | 135/1261 [00:26<03:52,  4.85it/s][A[A
    
     11%|█         | 136/1261 [00:27<03:56,  4.77it/s][A[A
    
     11%|█         | 137/1261 [00:27<03:57,  4.73it/s][A[A
    
     11%|█         | 138/1261 [00:27<03:46,  4.96it/s][A[A
    
     11%|█         | 139/1261 [00:27<03:49,  4.89it/s][A[A
    
     11%|█         | 140/1261 [00:27<03:43,  5.01it/s][A[A
    
     11%|█         | 141/1261 [00:28<03:47,  4.92it/s][A[A
    
     11%|█▏        | 142/1261 [00:28<03:49,  4.88it/s][A[A
    
     11%|█▏        | 143/1261 [00:28<03:49,  4.88it/s][A[A
    
     11%|█▏        | 144/1261 [00:28<03:46,  4.92it/s][A[A
    
     11%|█▏        | 145/1261 [00:29<03:52,  4.80it/s][A[A
    
     12%|█▏        | 146/1261 [00:29<03:53,  4.78it/s][A[A
    
     12%|█▏        | 147/1261 [00:29<03:53,  4.77it/s][A[A
    
     12%|█▏        | 148/1261 [00:29<03:51,  4.81it/s][A[A
    
     12%|█▏        | 149/1261 [00:29<03:52,  4.79it/s][A[A
    
     12%|█▏        | 150/1261 [00:30<03:53,  4.76it/s][A[A
    
     12%|█▏        | 151/1261 [00:30<03:57,  4.67it/s][A[A
    
     12%|█▏        | 152/1261 [00:30<03:55,  4.71it/s][A[A
    
     12%|█▏        | 153/1261 [00:30<03:53,  4.75it/s][A[A
    
     12%|█▏        | 154/1261 [00:30<03:57,  4.65it/s][A[A
    
     12%|█▏        | 155/1261 [00:31<03:55,  4.69it/s][A[A
    
     12%|█▏        | 156/1261 [00:31<03:55,  4.70it/s][A[A
    
     12%|█▏        | 157/1261 [00:31<03:55,  4.68it/s][A[A
    
     13%|█▎        | 158/1261 [00:31<03:55,  4.69it/s][A[A
    
     13%|█▎        | 159/1261 [00:31<03:50,  4.77it/s][A[A
    
     13%|█▎        | 160/1261 [00:32<03:52,  4.74it/s][A[A
    
     13%|█▎        | 161/1261 [00:32<03:52,  4.73it/s][A[A
    
     13%|█▎        | 162/1261 [00:32<03:52,  4.74it/s][A[A
    
     13%|█▎        | 163/1261 [00:32<03:51,  4.75it/s][A[A
    
     13%|█▎        | 164/1261 [00:33<03:51,  4.74it/s][A[A
    
     13%|█▎        | 165/1261 [00:33<03:46,  4.84it/s][A[A
    
     13%|█▎        | 166/1261 [00:33<03:42,  4.93it/s][A[A
    
     13%|█▎        | 167/1261 [00:33<03:36,  5.04it/s][A[A
    
     13%|█▎        | 168/1261 [00:33<03:37,  5.02it/s][A[A
    
     13%|█▎        | 169/1261 [00:34<03:36,  5.04it/s][A[A
    
     13%|█▎        | 170/1261 [00:34<03:36,  5.05it/s][A[A
    
     14%|█▎        | 171/1261 [00:34<03:32,  5.14it/s][A[A
    
     14%|█▎        | 172/1261 [00:34<03:35,  5.07it/s][A[A
    
     14%|█▎        | 173/1261 [00:34<03:34,  5.07it/s][A[A
    
     14%|█▍        | 174/1261 [00:34<03:33,  5.08it/s][A[A
    
     14%|█▍        | 175/1261 [00:35<03:36,  5.02it/s][A[A
    
     14%|█▍        | 176/1261 [00:35<03:39,  4.94it/s][A[A
    
     14%|█▍        | 177/1261 [00:35<03:35,  5.04it/s][A[A
    
     14%|█▍        | 178/1261 [00:35<03:33,  5.06it/s][A[A
    
     14%|█▍        | 179/1261 [00:36<03:39,  4.93it/s][A[A
    
     14%|█▍        | 180/1261 [00:36<03:45,  4.80it/s][A[A
    
     14%|█▍        | 181/1261 [00:36<03:40,  4.91it/s][A[A
    
     14%|█▍        | 182/1261 [00:36<03:43,  4.83it/s][A[A
    
     15%|█▍        | 183/1261 [00:36<03:44,  4.80it/s][A[A
    
     15%|█▍        | 184/1261 [00:37<03:40,  4.89it/s][A[A
    
     15%|█▍        | 185/1261 [00:37<03:34,  5.00it/s][A[A
    
     15%|█▍        | 186/1261 [00:37<03:32,  5.07it/s][A[A
    
     15%|█▍        | 187/1261 [00:37<03:30,  5.11it/s][A[A
    
     15%|█▍        | 188/1261 [00:37<03:27,  5.17it/s][A[A
    
     15%|█▍        | 189/1261 [00:37<03:27,  5.17it/s][A[A
    
     15%|█▌        | 190/1261 [00:38<03:29,  5.11it/s][A[A
    
     15%|█▌        | 191/1261 [00:38<03:31,  5.07it/s][A[A
    
     15%|█▌        | 192/1261 [00:38<03:32,  5.02it/s][A[A
    
     15%|█▌        | 193/1261 [00:38<03:30,  5.08it/s][A[A
    
     15%|█▌        | 194/1261 [00:38<03:29,  5.10it/s][A[A
    
     15%|█▌        | 195/1261 [00:39<03:26,  5.17it/s][A[A
    
     16%|█▌        | 196/1261 [00:39<03:25,  5.17it/s][A[A
    
     16%|█▌        | 197/1261 [00:39<03:24,  5.21it/s][A[A
    
     16%|█▌        | 198/1261 [00:39<03:23,  5.22it/s][A[A
    
     16%|█▌        | 199/1261 [00:39<03:21,  5.26it/s][A[A
    
     16%|█▌        | 200/1261 [00:40<03:29,  5.07it/s][A[A
    
     16%|█▌        | 201/1261 [00:40<03:26,  5.13it/s][A[A
    
     16%|█▌        | 202/1261 [00:40<03:24,  5.18it/s][A[A
    
     16%|█▌        | 203/1261 [00:40<03:24,  5.18it/s][A[A
    
     16%|█▌        | 204/1261 [00:40<03:30,  5.01it/s][A[A
    
     16%|█▋        | 205/1261 [00:41<03:30,  5.01it/s][A[A
    
     16%|█▋        | 206/1261 [00:41<03:32,  4.97it/s][A[A
    
     16%|█▋        | 207/1261 [00:41<03:33,  4.94it/s][A[A
    
     16%|█▋        | 208/1261 [00:41<03:33,  4.94it/s][A[A
    
     17%|█▋        | 209/1261 [00:41<03:34,  4.91it/s][A[A
    
     17%|█▋        | 210/1261 [00:42<03:32,  4.95it/s][A[A
    
     17%|█▋        | 211/1261 [00:42<03:35,  4.88it/s][A[A
    
     17%|█▋        | 212/1261 [00:42<03:36,  4.84it/s][A[A
    
     17%|█▋        | 213/1261 [00:42<03:35,  4.87it/s][A[A
    
     17%|█▋        | 214/1261 [00:42<03:34,  4.88it/s][A[A
    
     17%|█▋        | 215/1261 [00:43<03:36,  4.82it/s][A[A
    
     17%|█▋        | 216/1261 [00:43<03:38,  4.78it/s][A[A
    
     17%|█▋        | 217/1261 [00:43<03:30,  4.95it/s][A[A
    
     17%|█▋        | 218/1261 [00:43<03:37,  4.79it/s][A[A
    
     17%|█▋        | 219/1261 [00:44<03:35,  4.84it/s][A[A
    
     17%|█▋        | 220/1261 [00:44<03:36,  4.81it/s][A[A
    
     18%|█▊        | 221/1261 [00:44<03:28,  4.98it/s][A[A
    
     18%|█▊        | 222/1261 [00:44<03:32,  4.90it/s][A[A
    
     18%|█▊        | 223/1261 [00:44<03:32,  4.88it/s][A[A
    
     18%|█▊        | 224/1261 [00:45<03:32,  4.88it/s][A[A
    
     18%|█▊        | 225/1261 [00:45<03:38,  4.75it/s][A[A
    
     18%|█▊        | 226/1261 [00:45<03:36,  4.77it/s][A[A
    
     18%|█▊        | 227/1261 [00:45<03:31,  4.89it/s][A[A
    
     18%|█▊        | 228/1261 [00:45<03:28,  4.96it/s][A[A
    
     18%|█▊        | 229/1261 [00:46<03:25,  5.01it/s][A[A
    
     18%|█▊        | 230/1261 [00:46<03:28,  4.95it/s][A[A
    
     18%|█▊        | 231/1261 [00:46<03:29,  4.92it/s][A[A
    
     18%|█▊        | 232/1261 [00:46<03:29,  4.91it/s][A[A
    
     18%|█▊        | 233/1261 [00:46<03:34,  4.80it/s][A[A
    
     19%|█▊        | 234/1261 [00:47<03:31,  4.86it/s][A[A
    
     19%|█▊        | 235/1261 [00:47<03:31,  4.84it/s][A[A
    
     19%|█▊        | 236/1261 [00:47<03:33,  4.80it/s][A[A
    
     19%|█▉        | 237/1261 [00:47<03:32,  4.81it/s][A[A
    
     19%|█▉        | 238/1261 [00:47<03:32,  4.82it/s][A[A
    
     19%|█▉        | 239/1261 [00:48<03:31,  4.84it/s][A[A
    
     19%|█▉        | 240/1261 [00:48<03:31,  4.82it/s][A[A
    
     19%|█▉        | 241/1261 [00:48<03:32,  4.81it/s][A[A
    
     19%|█▉        | 242/1261 [00:48<03:32,  4.79it/s][A[A
    
     19%|█▉        | 243/1261 [00:48<03:34,  4.75it/s][A[A
    
     19%|█▉        | 244/1261 [00:49<03:35,  4.72it/s][A[A
    
     19%|█▉        | 245/1261 [00:49<03:34,  4.73it/s][A[A
    
     20%|█▉        | 246/1261 [00:49<03:37,  4.66it/s][A[A
    
     20%|█▉        | 247/1261 [00:49<03:37,  4.67it/s][A[A
    
     20%|█▉        | 248/1261 [00:50<03:37,  4.66it/s][A[A
    
     20%|█▉        | 249/1261 [00:50<03:36,  4.67it/s][A[A
    
     20%|█▉        | 250/1261 [00:50<03:37,  4.65it/s][A[A
    
     20%|█▉        | 251/1261 [00:50<03:32,  4.75it/s][A[A
    
     20%|█▉        | 252/1261 [00:50<03:31,  4.76it/s][A[A
    
     20%|██        | 253/1261 [00:51<03:29,  4.82it/s][A[A
    
     20%|██        | 254/1261 [00:51<03:41,  4.55it/s][A[A
    
     20%|██        | 255/1261 [00:51<03:50,  4.36it/s][A[A
    
     20%|██        | 256/1261 [00:51<03:38,  4.61it/s][A[A
    
     20%|██        | 257/1261 [00:51<03:36,  4.65it/s][A[A
    
     20%|██        | 258/1261 [00:52<03:36,  4.64it/s][A[A
    
     21%|██        | 259/1261 [00:52<03:32,  4.71it/s][A[A
    
     21%|██        | 260/1261 [00:52<03:31,  4.73it/s][A[A
    
     21%|██        | 261/1261 [00:52<03:30,  4.74it/s][A[A
    
     21%|██        | 262/1261 [00:53<03:25,  4.86it/s][A[A
    
     21%|██        | 263/1261 [00:53<03:19,  4.99it/s][A[A
    
     21%|██        | 264/1261 [00:53<03:22,  4.93it/s][A[A
    
     21%|██        | 265/1261 [00:53<03:25,  4.85it/s][A[A
    
     21%|██        | 266/1261 [00:53<03:19,  4.98it/s][A[A
    
     21%|██        | 267/1261 [00:54<03:16,  5.06it/s][A[A
    
     21%|██▏       | 268/1261 [00:54<03:15,  5.09it/s][A[A
    
     21%|██▏       | 269/1261 [00:54<03:16,  5.05it/s][A[A
    
     21%|██▏       | 270/1261 [00:54<03:15,  5.08it/s][A[A
    
     21%|██▏       | 271/1261 [00:54<03:14,  5.09it/s][A[A
    
     22%|██▏       | 272/1261 [00:55<03:19,  4.95it/s][A[A
    
     22%|██▏       | 273/1261 [00:55<03:19,  4.96it/s][A[A
    
     22%|██▏       | 274/1261 [00:55<03:18,  4.97it/s][A[A
    
     22%|██▏       | 275/1261 [00:55<03:19,  4.95it/s][A[A
    
     22%|██▏       | 276/1261 [00:55<03:18,  4.97it/s][A[A
    
     22%|██▏       | 277/1261 [00:56<03:19,  4.92it/s][A[A
    
     22%|██▏       | 278/1261 [00:56<03:17,  4.97it/s][A[A
    
     22%|██▏       | 279/1261 [00:56<03:21,  4.88it/s][A[A
    
     22%|██▏       | 280/1261 [00:56<03:25,  4.77it/s][A[A
    
     22%|██▏       | 281/1261 [00:56<03:22,  4.85it/s][A[A
    
     22%|██▏       | 282/1261 [00:57<03:22,  4.84it/s][A[A
    
     22%|██▏       | 283/1261 [00:57<03:24,  4.79it/s][A[A
    
     23%|██▎       | 284/1261 [00:57<03:25,  4.76it/s][A[A
    
     23%|██▎       | 285/1261 [00:57<03:25,  4.74it/s][A[A
    
     23%|██▎       | 286/1261 [00:57<03:25,  4.75it/s][A[A
    
     23%|██▎       | 287/1261 [00:58<03:22,  4.82it/s][A[A
    
     23%|██▎       | 288/1261 [00:58<03:23,  4.79it/s][A[A
    
     23%|██▎       | 289/1261 [00:58<03:22,  4.80it/s][A[A
    
     23%|██▎       | 290/1261 [00:58<03:22,  4.80it/s][A[A
    
     23%|██▎       | 291/1261 [00:58<03:26,  4.71it/s][A[A
    
     23%|██▎       | 292/1261 [00:59<03:19,  4.86it/s][A[A
    
     23%|██▎       | 293/1261 [00:59<03:40,  4.40it/s][A[A
    
     23%|██▎       | 294/1261 [00:59<03:29,  4.62it/s][A[A
    
     23%|██▎       | 295/1261 [00:59<03:23,  4.75it/s][A[A
    
     23%|██▎       | 296/1261 [00:59<03:16,  4.92it/s][A[A
    
     24%|██▎       | 297/1261 [01:00<03:14,  4.96it/s][A[A
    
     24%|██▎       | 298/1261 [01:00<03:13,  4.97it/s][A[A
    
     24%|██▎       | 299/1261 [01:00<03:10,  5.04it/s][A[A
    
     24%|██▍       | 300/1261 [01:00<03:13,  4.97it/s][A[A
    
     24%|██▍       | 301/1261 [01:00<03:07,  5.11it/s][A[A
    
     24%|██▍       | 302/1261 [01:01<03:12,  4.97it/s][A[A
    
     24%|██▍       | 303/1261 [01:01<03:15,  4.90it/s][A[A
    
     24%|██▍       | 304/1261 [01:01<03:10,  5.03it/s][A[A
    
     24%|██▍       | 305/1261 [01:01<03:07,  5.11it/s][A[A
    
     24%|██▍       | 306/1261 [01:01<03:03,  5.20it/s][A[A
    
     24%|██▍       | 307/1261 [01:02<03:01,  5.25it/s][A[A
    
     24%|██▍       | 308/1261 [01:02<02:59,  5.30it/s][A[A
    
     25%|██▍       | 309/1261 [01:02<03:06,  5.11it/s][A[A
    
     25%|██▍       | 310/1261 [01:02<03:09,  5.02it/s][A[A
    
     25%|██▍       | 311/1261 [01:02<03:12,  4.93it/s][A[A
    
     25%|██▍       | 312/1261 [01:03<03:16,  4.82it/s][A[A
    
     25%|██▍       | 313/1261 [01:03<03:15,  4.85it/s][A[A
    
     25%|██▍       | 314/1261 [01:03<03:14,  4.88it/s][A[A
    
     25%|██▍       | 315/1261 [01:03<03:16,  4.82it/s][A[A
    
     25%|██▌       | 316/1261 [01:04<03:13,  4.88it/s][A[A
    
     25%|██▌       | 317/1261 [01:04<03:11,  4.92it/s][A[A
    
     25%|██▌       | 318/1261 [01:04<03:15,  4.83it/s][A[A
    
     25%|██▌       | 319/1261 [01:04<03:14,  4.84it/s][A[A
    
     25%|██▌       | 320/1261 [01:04<03:18,  4.74it/s][A[A
    
     25%|██▌       | 321/1261 [01:05<03:22,  4.64it/s][A[A
    
     26%|██▌       | 322/1261 [01:05<03:15,  4.79it/s][A[A
    
     26%|██▌       | 323/1261 [01:05<03:18,  4.72it/s][A[A
    
     26%|██▌       | 324/1261 [01:05<03:15,  4.80it/s][A[A
    
     26%|██▌       | 325/1261 [01:05<03:15,  4.78it/s][A[A
    
     26%|██▌       | 326/1261 [01:06<03:19,  4.70it/s][A[A
    
     26%|██▌       | 327/1261 [01:06<03:16,  4.76it/s][A[A
    
     26%|██▌       | 328/1261 [01:06<03:12,  4.84it/s][A[A
    
     26%|██▌       | 329/1261 [01:06<03:12,  4.85it/s][A[A
    
     26%|██▌       | 330/1261 [01:06<03:06,  5.00it/s][A[A
    
     26%|██▌       | 331/1261 [01:07<03:03,  5.06it/s][A[A
    
     26%|██▋       | 332/1261 [01:07<02:59,  5.18it/s][A[A
    
     26%|██▋       | 333/1261 [01:07<02:58,  5.20it/s][A[A
    
     26%|██▋       | 334/1261 [01:07<02:56,  5.27it/s][A[A
    
     27%|██▋       | 335/1261 [01:07<02:54,  5.31it/s][A[A
    
     27%|██▋       | 336/1261 [01:08<02:55,  5.26it/s][A[A
    
     27%|██▋       | 337/1261 [01:08<02:54,  5.28it/s][A[A
    
     27%|██▋       | 338/1261 [01:08<03:04,  5.00it/s][A[A
    
     27%|██▋       | 339/1261 [01:08<03:06,  4.95it/s][A[A
    
     27%|██▋       | 340/1261 [01:08<03:01,  5.09it/s][A[A
    
     27%|██▋       | 341/1261 [01:09<02:57,  5.17it/s][A[A
    
     27%|██▋       | 342/1261 [01:09<02:55,  5.25it/s][A[A
    
     27%|██▋       | 343/1261 [01:09<02:55,  5.24it/s][A[A
    
     27%|██▋       | 344/1261 [01:09<02:54,  5.26it/s][A[A
    
     27%|██▋       | 345/1261 [01:09<02:52,  5.30it/s][A[A
    
     27%|██▋       | 346/1261 [01:09<02:54,  5.23it/s][A[A
    
     28%|██▊       | 347/1261 [01:10<02:59,  5.10it/s][A[A
    
     28%|██▊       | 348/1261 [01:10<03:00,  5.06it/s][A[A
    
     28%|██▊       | 349/1261 [01:10<03:01,  5.01it/s][A[A
    
     28%|██▊       | 350/1261 [01:10<03:03,  4.96it/s][A[A
    
     28%|██▊       | 351/1261 [01:10<03:02,  4.99it/s][A[A
    
     28%|██▊       | 352/1261 [01:11<03:02,  4.97it/s][A[A
    
     28%|██▊       | 353/1261 [01:11<02:58,  5.08it/s][A[A
    
     28%|██▊       | 354/1261 [01:11<02:55,  5.18it/s][A[A
    
     28%|██▊       | 355/1261 [01:11<02:53,  5.22it/s][A[A
    
     28%|██▊       | 356/1261 [01:11<02:57,  5.09it/s][A[A
    
     28%|██▊       | 357/1261 [01:12<02:57,  5.10it/s][A[A
    
     28%|██▊       | 358/1261 [01:12<02:58,  5.06it/s][A[A
    
     28%|██▊       | 359/1261 [01:12<02:54,  5.17it/s][A[A
    
     29%|██▊       | 360/1261 [01:12<02:52,  5.23it/s][A[A
    
     29%|██▊       | 361/1261 [01:12<02:50,  5.29it/s][A[A
    
     29%|██▊       | 362/1261 [01:13<02:55,  5.12it/s][A[A
    
     29%|██▉       | 363/1261 [01:13<02:55,  5.11it/s][A[A
    
     29%|██▉       | 364/1261 [01:13<02:58,  5.02it/s][A[A
    
     29%|██▉       | 365/1261 [01:13<02:59,  5.00it/s][A[A
    
     29%|██▉       | 366/1261 [01:13<03:00,  4.97it/s][A[A
    
     29%|██▉       | 367/1261 [01:14<03:06,  4.79it/s][A[A
    
     29%|██▉       | 368/1261 [01:14<03:03,  4.86it/s][A[A
    
     29%|██▉       | 369/1261 [01:14<02:58,  4.99it/s][A[A
    
     29%|██▉       | 370/1261 [01:14<02:54,  5.12it/s][A[A
    
     29%|██▉       | 371/1261 [01:14<02:52,  5.16it/s][A[A
    
     30%|██▉       | 372/1261 [01:15<02:50,  5.21it/s][A[A
    
     30%|██▉       | 373/1261 [01:15<02:49,  5.25it/s][A[A
    
     30%|██▉       | 374/1261 [01:15<02:48,  5.26it/s][A[A
    
     30%|██▉       | 375/1261 [01:15<02:47,  5.27it/s][A[A
    
     30%|██▉       | 376/1261 [01:15<02:47,  5.30it/s][A[A
    
     30%|██▉       | 377/1261 [01:16<02:46,  5.30it/s][A[A
    
     30%|██▉       | 378/1261 [01:16<02:46,  5.30it/s][A[A
    
     30%|███       | 379/1261 [01:16<02:45,  5.33it/s][A[A
    
     30%|███       | 380/1261 [01:16<02:44,  5.34it/s][A[A
    
     30%|███       | 381/1261 [01:16<02:43,  5.37it/s][A[A
    
     30%|███       | 382/1261 [01:16<02:44,  5.33it/s][A[A
    
     30%|███       | 383/1261 [01:17<02:45,  5.31it/s][A[A
    
     30%|███       | 384/1261 [01:17<02:45,  5.28it/s][A[A
    
     31%|███       | 385/1261 [01:17<02:44,  5.33it/s][A[A
    
     31%|███       | 386/1261 [01:17<02:44,  5.33it/s][A[A
    
     31%|███       | 387/1261 [01:17<02:43,  5.34it/s][A[A
    
     31%|███       | 388/1261 [01:18<02:44,  5.32it/s][A[A
    
     31%|███       | 389/1261 [01:18<02:42,  5.35it/s][A[A
    
     31%|███       | 390/1261 [01:18<02:43,  5.33it/s][A[A
    
     31%|███       | 391/1261 [01:18<02:42,  5.34it/s][A[A
    
     31%|███       | 392/1261 [01:18<02:43,  5.31it/s][A[A
    
     31%|███       | 393/1261 [01:19<02:45,  5.23it/s][A[A
    
     31%|███       | 394/1261 [01:19<02:44,  5.26it/s][A[A
    
     31%|███▏      | 395/1261 [01:19<02:42,  5.31it/s][A[A
    
     31%|███▏      | 396/1261 [01:19<02:41,  5.35it/s][A[A
    
     31%|███▏      | 397/1261 [01:19<02:41,  5.34it/s][A[A
    
     32%|███▏      | 398/1261 [01:20<02:48,  5.11it/s][A[A
    
     32%|███▏      | 399/1261 [01:20<02:51,  5.02it/s][A[A
    
     32%|███▏      | 400/1261 [01:20<02:51,  5.01it/s][A[A
    
     32%|███▏      | 401/1261 [01:20<02:52,  4.99it/s][A[A
    
     32%|███▏      | 402/1261 [01:20<02:55,  4.90it/s][A[A
    
     32%|███▏      | 403/1261 [01:21<02:57,  4.84it/s][A[A
    
     32%|███▏      | 404/1261 [01:21<02:52,  4.96it/s][A[A
    
     32%|███▏      | 405/1261 [01:21<02:48,  5.07it/s][A[A
    
     32%|███▏      | 406/1261 [01:21<02:45,  5.16it/s][A[A
    
     32%|███▏      | 407/1261 [01:21<02:42,  5.25it/s][A[A
    
     32%|███▏      | 408/1261 [01:22<02:46,  5.11it/s][A[A
    
     32%|███▏      | 409/1261 [01:22<02:43,  5.21it/s][A[A
    
     33%|███▎      | 410/1261 [01:22<02:43,  5.20it/s][A[A
    
     33%|███▎      | 411/1261 [01:22<02:41,  5.26it/s][A[A
    
     33%|███▎      | 412/1261 [01:22<02:39,  5.31it/s][A[A
    
     33%|███▎      | 413/1261 [01:22<02:41,  5.25it/s][A[A
    
     33%|███▎      | 414/1261 [01:23<02:40,  5.28it/s][A[A
    
     33%|███▎      | 415/1261 [01:23<02:39,  5.29it/s][A[A
    
     33%|███▎      | 416/1261 [01:23<02:39,  5.30it/s][A[A
    
     33%|███▎      | 417/1261 [01:23<02:39,  5.31it/s][A[A
    
     33%|███▎      | 418/1261 [01:23<02:44,  5.11it/s][A[A
    
     33%|███▎      | 419/1261 [01:24<02:42,  5.17it/s][A[A
    
     33%|███▎      | 420/1261 [01:24<02:40,  5.22it/s][A[A
    
     33%|███▎      | 421/1261 [01:24<02:39,  5.27it/s][A[A
    
     33%|███▎      | 422/1261 [01:24<02:38,  5.28it/s][A[A
    
     34%|███▎      | 423/1261 [01:24<02:44,  5.08it/s][A[A
    
     34%|███▎      | 424/1261 [01:25<02:44,  5.10it/s][A[A
    
     34%|███▎      | 425/1261 [01:25<02:41,  5.19it/s][A[A
    
     34%|███▍      | 426/1261 [01:25<02:38,  5.26it/s][A[A
    
     34%|███▍      | 427/1261 [01:25<02:39,  5.23it/s][A[A
    
     34%|███▍      | 428/1261 [01:25<02:38,  5.24it/s][A[A
    
     34%|███▍      | 429/1261 [01:26<02:38,  5.26it/s][A[A
    
     34%|███▍      | 430/1261 [01:26<02:37,  5.28it/s][A[A
    
     34%|███▍      | 431/1261 [01:26<02:36,  5.32it/s][A[A
    
     34%|███▍      | 432/1261 [01:26<02:36,  5.29it/s][A[A
    
     34%|███▍      | 433/1261 [01:26<02:36,  5.29it/s][A[A
    
     34%|███▍      | 434/1261 [01:26<02:36,  5.28it/s][A[A
    
     34%|███▍      | 435/1261 [01:27<02:37,  5.23it/s][A[A
    
     35%|███▍      | 436/1261 [01:27<02:37,  5.25it/s][A[A
    
     35%|███▍      | 437/1261 [01:27<02:36,  5.28it/s][A[A
    
     35%|███▍      | 438/1261 [01:27<02:36,  5.26it/s][A[A
    
     35%|███▍      | 439/1261 [01:27<02:34,  5.30it/s][A[A
    
     35%|███▍      | 440/1261 [01:28<02:39,  5.14it/s][A[A
    
     35%|███▍      | 441/1261 [01:28<02:41,  5.09it/s][A[A
    
     35%|███▌      | 442/1261 [01:28<02:42,  5.03it/s][A[A
    
     35%|███▌      | 443/1261 [01:28<02:44,  4.97it/s][A[A
    
     35%|███▌      | 444/1261 [01:28<02:45,  4.94it/s][A[A
    
     35%|███▌      | 445/1261 [01:29<02:44,  4.95it/s][A[A
    
     35%|███▌      | 446/1261 [01:29<02:43,  4.98it/s][A[A
    
     35%|███▌      | 447/1261 [01:29<02:43,  4.97it/s][A[A
    
     36%|███▌      | 448/1261 [01:29<02:43,  4.96it/s][A[A
    
     36%|███▌      | 449/1261 [01:29<02:44,  4.95it/s][A[A
    
     36%|███▌      | 450/1261 [01:30<02:43,  4.97it/s][A[A
    
     36%|███▌      | 451/1261 [01:30<02:42,  4.99it/s][A[A
    
     36%|███▌      | 452/1261 [01:30<02:41,  5.00it/s][A[A
    
     36%|███▌      | 453/1261 [01:30<02:42,  4.96it/s][A[A
    
     36%|███▌      | 454/1261 [01:30<02:40,  5.03it/s][A[A
    
     36%|███▌      | 455/1261 [01:31<02:42,  4.97it/s][A[A
    
     36%|███▌      | 456/1261 [01:31<02:42,  4.96it/s][A[A
    
     36%|███▌      | 457/1261 [01:31<02:48,  4.78it/s][A[A
    
     36%|███▋      | 458/1261 [01:31<02:44,  4.88it/s][A[A
    
     36%|███▋      | 459/1261 [01:31<02:42,  4.94it/s][A[A
    
     36%|███▋      | 460/1261 [01:32<02:39,  5.04it/s][A[A
    
     37%|███▋      | 461/1261 [01:32<02:38,  5.05it/s][A[A
    
     37%|███▋      | 462/1261 [01:32<02:35,  5.13it/s][A[A
    
     37%|███▋      | 463/1261 [01:32<02:36,  5.11it/s][A[A
    
     37%|███▋      | 464/1261 [01:32<02:39,  4.99it/s][A[A
    
     37%|███▋      | 465/1261 [01:33<02:44,  4.83it/s][A[A
    
     37%|███▋      | 466/1261 [01:33<02:40,  4.96it/s][A[A
    
     37%|███▋      | 467/1261 [01:33<02:36,  5.07it/s][A[A
    
     37%|███▋      | 468/1261 [01:33<02:40,  4.93it/s][A[A
    
     37%|███▋      | 469/1261 [01:33<02:37,  5.03it/s][A[A
    
     37%|███▋      | 470/1261 [01:34<02:36,  5.06it/s][A[A
    
     37%|███▋      | 471/1261 [01:34<02:40,  4.93it/s][A[A
    
     37%|███▋      | 472/1261 [01:34<02:35,  5.08it/s][A[A
    
     38%|███▊      | 473/1261 [01:34<02:37,  5.01it/s][A[A
    
     38%|███▊      | 474/1261 [01:34<02:34,  5.09it/s][A[A
    
     38%|███▊      | 475/1261 [01:35<02:36,  5.01it/s][A[A
    
     38%|███▊      | 476/1261 [01:35<02:39,  4.93it/s][A[A
    
     38%|███▊      | 477/1261 [01:35<02:34,  5.08it/s][A[A
    
     38%|███▊      | 478/1261 [01:35<02:31,  5.17it/s][A[A
    
     38%|███▊      | 479/1261 [01:35<02:30,  5.19it/s][A[A
    
     38%|███▊      | 480/1261 [01:36<02:28,  5.24it/s][A[A
    
     38%|███▊      | 481/1261 [01:36<02:27,  5.29it/s][A[A
    
     38%|███▊      | 482/1261 [01:36<02:30,  5.17it/s][A[A
    
     38%|███▊      | 483/1261 [01:36<02:27,  5.27it/s][A[A
    
     38%|███▊      | 484/1261 [01:36<02:27,  5.27it/s][A[A
    
     38%|███▊      | 485/1261 [01:37<02:27,  5.27it/s][A[A
    
     39%|███▊      | 486/1261 [01:37<02:26,  5.29it/s][A[A
    
     39%|███▊      | 487/1261 [01:37<02:26,  5.27it/s][A[A
    
     39%|███▊      | 488/1261 [01:37<02:26,  5.28it/s][A[A
    
     39%|███▉      | 489/1261 [01:37<02:26,  5.28it/s][A[A
    
     39%|███▉      | 490/1261 [01:37<02:26,  5.27it/s][A[A
    
     39%|███▉      | 491/1261 [01:38<02:26,  5.27it/s][A[A
    
     39%|███▉      | 492/1261 [01:38<02:26,  5.26it/s][A[A
    
     39%|███▉      | 493/1261 [01:38<02:26,  5.23it/s][A[A
    
     39%|███▉      | 494/1261 [01:38<02:24,  5.29it/s][A[A
    
     39%|███▉      | 495/1261 [01:38<02:23,  5.35it/s][A[A
    
     39%|███▉      | 496/1261 [01:39<02:28,  5.15it/s][A[A
    
     39%|███▉      | 497/1261 [01:39<02:28,  5.16it/s][A[A
    
     39%|███▉      | 498/1261 [01:39<02:26,  5.20it/s][A[A
    
     40%|███▉      | 499/1261 [01:39<02:26,  5.22it/s][A[A
    
     40%|███▉      | 500/1261 [01:39<02:27,  5.15it/s][A[A
    
     40%|███▉      | 501/1261 [01:40<02:29,  5.07it/s][A[A
    
     40%|███▉      | 502/1261 [01:40<02:31,  5.01it/s][A[A
    
     40%|███▉      | 503/1261 [01:40<02:32,  4.97it/s][A[A
    
     40%|███▉      | 504/1261 [01:40<02:28,  5.10it/s][A[A
    
     40%|████      | 505/1261 [01:40<02:29,  5.06it/s][A[A
    
     40%|████      | 506/1261 [01:41<02:29,  5.06it/s][A[A
    
     40%|████      | 507/1261 [01:41<02:30,  5.02it/s][A[A
    
     40%|████      | 508/1261 [01:41<02:27,  5.11it/s][A[A
    
     40%|████      | 509/1261 [01:41<02:27,  5.09it/s][A[A
    
     40%|████      | 510/1261 [01:41<02:23,  5.22it/s][A[A
    
     41%|████      | 511/1261 [01:42<02:27,  5.09it/s][A[A
    
     41%|████      | 512/1261 [01:42<02:28,  5.04it/s][A[A
    
     41%|████      | 513/1261 [01:42<02:28,  5.05it/s][A[A
    
     41%|████      | 514/1261 [01:42<02:24,  5.16it/s][A[A
    
     41%|████      | 515/1261 [01:42<02:26,  5.11it/s][A[A
    
     41%|████      | 516/1261 [01:43<02:26,  5.10it/s][A[A
    
     41%|████      | 517/1261 [01:43<02:29,  4.99it/s][A[A
    
     41%|████      | 518/1261 [01:43<02:26,  5.06it/s][A[A
    
     41%|████      | 519/1261 [01:43<02:24,  5.14it/s][A[A
    
     41%|████      | 520/1261 [01:43<02:23,  5.17it/s][A[A
    
     41%|████▏     | 521/1261 [01:44<02:26,  5.06it/s][A[A
    
     41%|████▏     | 522/1261 [01:44<02:23,  5.16it/s][A[A
    
     41%|████▏     | 523/1261 [01:44<02:23,  5.15it/s][A[A
    
     42%|████▏     | 524/1261 [01:44<02:22,  5.18it/s][A[A
    
     42%|████▏     | 525/1261 [01:44<02:20,  5.23it/s][A[A
    
     42%|████▏     | 526/1261 [01:45<02:21,  5.21it/s][A[A
    
     42%|████▏     | 527/1261 [01:45<02:21,  5.20it/s][A[A
    
     42%|████▏     | 528/1261 [01:45<02:21,  5.17it/s][A[A
    
     42%|████▏     | 529/1261 [01:45<02:19,  5.23it/s][A[A
    
     42%|████▏     | 530/1261 [01:45<02:23,  5.08it/s][A[A
    
     42%|████▏     | 531/1261 [01:45<02:25,  5.03it/s][A[A
    
     42%|████▏     | 532/1261 [01:46<02:55,  4.15it/s][A[A
    
     42%|████▏     | 533/1261 [01:46<02:50,  4.27it/s][A[A
    
     42%|████▏     | 534/1261 [01:46<02:43,  4.45it/s][A[A
    
     42%|████▏     | 535/1261 [01:46<02:40,  4.52it/s][A[A
    
     43%|████▎     | 536/1261 [01:47<02:38,  4.57it/s][A[A
    
     43%|████▎     | 537/1261 [01:47<02:32,  4.74it/s][A[A
    
     43%|████▎     | 538/1261 [01:47<02:31,  4.77it/s][A[A
    
     43%|████▎     | 539/1261 [01:47<02:34,  4.68it/s][A[A
    
     43%|████▎     | 540/1261 [01:48<02:32,  4.72it/s][A[A
    
     43%|████▎     | 541/1261 [01:48<02:28,  4.86it/s][A[A
    
     43%|████▎     | 542/1261 [01:48<02:24,  4.99it/s][A[A
    
     43%|████▎     | 543/1261 [01:48<02:26,  4.89it/s][A[A
    
     43%|████▎     | 544/1261 [01:48<02:23,  5.00it/s][A[A
    
     43%|████▎     | 545/1261 [01:48<02:20,  5.08it/s][A[A
    
     43%|████▎     | 546/1261 [01:49<02:18,  5.17it/s][A[A
    
     43%|████▎     | 547/1261 [01:49<02:20,  5.08it/s][A[A
    
     43%|████▎     | 548/1261 [01:49<02:17,  5.18it/s][A[A
    
     44%|████▎     | 549/1261 [01:49<02:17,  5.17it/s][A[A
    
     44%|████▎     | 550/1261 [01:49<02:18,  5.15it/s][A[A
    
     44%|████▎     | 551/1261 [01:50<02:16,  5.19it/s][A[A
    
     44%|████▍     | 552/1261 [01:50<02:20,  5.03it/s][A[A
    
     44%|████▍     | 553/1261 [01:50<02:18,  5.10it/s][A[A
    
     44%|████▍     | 554/1261 [01:50<02:17,  5.13it/s][A[A
    
     44%|████▍     | 555/1261 [01:50<02:16,  5.16it/s][A[A
    
     44%|████▍     | 556/1261 [01:51<02:15,  5.19it/s][A[A
    
     44%|████▍     | 557/1261 [01:51<02:14,  5.22it/s][A[A
    
     44%|████▍     | 558/1261 [01:51<02:18,  5.06it/s][A[A
    
     44%|████▍     | 559/1261 [01:51<02:22,  4.94it/s][A[A
    
     44%|████▍     | 560/1261 [01:51<02:19,  5.02it/s][A[A
    
     44%|████▍     | 561/1261 [01:52<02:19,  5.01it/s][A[A
    
     45%|████▍     | 562/1261 [01:52<02:19,  5.01it/s][A[A
    
     45%|████▍     | 563/1261 [01:52<02:23,  4.87it/s][A[A
    
     45%|████▍     | 564/1261 [01:52<02:23,  4.85it/s][A[A
    
     45%|████▍     | 565/1261 [01:52<02:23,  4.83it/s][A[A
    
     45%|████▍     | 566/1261 [01:53<02:22,  4.89it/s][A[A
    
     45%|████▍     | 567/1261 [01:53<02:22,  4.88it/s][A[A
    
     45%|████▌     | 568/1261 [01:53<02:18,  5.02it/s][A[A
    
     45%|████▌     | 569/1261 [01:53<02:22,  4.85it/s][A[A
    
     45%|████▌     | 570/1261 [01:54<02:35,  4.46it/s][A[A
    
     45%|████▌     | 571/1261 [01:54<02:30,  4.58it/s][A[A
    
     45%|████▌     | 572/1261 [01:54<02:26,  4.70it/s][A[A
    
     45%|████▌     | 573/1261 [01:54<02:27,  4.67it/s][A[A
    
     46%|████▌     | 574/1261 [01:54<02:23,  4.80it/s][A[A
    
     46%|████▌     | 575/1261 [01:55<02:22,  4.82it/s][A[A
    
     46%|████▌     | 576/1261 [01:55<02:21,  4.83it/s][A[A
    
     46%|████▌     | 577/1261 [01:55<02:22,  4.79it/s][A[A
    
     46%|████▌     | 578/1261 [01:55<02:22,  4.79it/s][A[A
    
     46%|████▌     | 579/1261 [01:55<02:20,  4.86it/s][A[A
    
     46%|████▌     | 580/1261 [01:56<02:21,  4.82it/s][A[A
    
     46%|████▌     | 581/1261 [01:56<02:22,  4.78it/s][A[A
    
     46%|████▌     | 582/1261 [01:56<02:20,  4.84it/s][A[A
    
     46%|████▌     | 583/1261 [01:56<02:19,  4.86it/s][A[A
    
     46%|████▋     | 584/1261 [01:56<02:19,  4.86it/s][A[A
    
     46%|████▋     | 585/1261 [01:57<02:17,  4.92it/s][A[A
    
     46%|████▋     | 586/1261 [01:57<02:18,  4.88it/s][A[A
    
     47%|████▋     | 587/1261 [01:57<02:19,  4.83it/s][A[A
    
     47%|████▋     | 588/1261 [01:57<02:17,  4.90it/s][A[A
    
     47%|████▋     | 589/1261 [01:57<02:15,  4.95it/s][A[A
    
     47%|████▋     | 590/1261 [01:58<02:15,  4.94it/s][A[A
    
     47%|████▋     | 591/1261 [01:58<02:16,  4.93it/s][A[A
    
     47%|████▋     | 592/1261 [01:58<02:16,  4.89it/s][A[A
    
     47%|████▋     | 593/1261 [01:58<02:16,  4.89it/s][A[A
    
     47%|████▋     | 594/1261 [01:58<02:12,  5.05it/s][A[A
    
     47%|████▋     | 595/1261 [01:59<02:10,  5.08it/s][A[A
    
     47%|████▋     | 596/1261 [01:59<02:12,  5.03it/s][A[A
    
     47%|████▋     | 597/1261 [01:59<02:08,  5.15it/s][A[A
    
     47%|████▋     | 598/1261 [01:59<02:08,  5.18it/s][A[A
    
     48%|████▊     | 599/1261 [01:59<02:08,  5.15it/s][A[A
    
     48%|████▊     | 600/1261 [02:00<02:08,  5.14it/s][A[A
    
     48%|████▊     | 601/1261 [02:00<02:10,  5.06it/s][A[A
    
     48%|████▊     | 602/1261 [02:00<02:09,  5.09it/s][A[A
    
     48%|████▊     | 603/1261 [02:00<02:09,  5.07it/s][A[A
    
     48%|████▊     | 604/1261 [02:00<02:12,  4.97it/s][A[A
    
     48%|████▊     | 605/1261 [02:01<02:12,  4.95it/s][A[A
    
     48%|████▊     | 606/1261 [02:01<02:10,  5.00it/s][A[A
    
     48%|████▊     | 607/1261 [02:01<02:12,  4.92it/s][A[A
    
     48%|████▊     | 608/1261 [02:01<02:11,  4.98it/s][A[A
    
     48%|████▊     | 609/1261 [02:01<02:10,  4.99it/s][A[A
    
     48%|████▊     | 610/1261 [02:02<02:10,  4.99it/s][A[A
    
     48%|████▊     | 611/1261 [02:02<02:09,  5.00it/s][A[A
    
     49%|████▊     | 612/1261 [02:02<02:10,  4.99it/s][A[A
    
     49%|████▊     | 613/1261 [02:02<02:09,  5.01it/s][A[A
    
     49%|████▊     | 614/1261 [02:02<02:10,  4.97it/s][A[A
    
     49%|████▉     | 615/1261 [02:03<02:10,  4.93it/s][A[A
    
     49%|████▉     | 616/1261 [02:03<02:09,  4.97it/s][A[A
    
     49%|████▉     | 617/1261 [02:03<02:10,  4.94it/s][A[A
    
     49%|████▉     | 618/1261 [02:03<02:09,  4.96it/s][A[A
    
     49%|████▉     | 619/1261 [02:03<02:10,  4.90it/s][A[A
    
     49%|████▉     | 620/1261 [02:04<02:10,  4.92it/s][A[A
    
     49%|████▉     | 621/1261 [02:04<02:09,  4.93it/s][A[A
    
     49%|████▉     | 622/1261 [02:04<02:09,  4.92it/s][A[A
    
     49%|████▉     | 623/1261 [02:04<02:10,  4.90it/s][A[A
    
     49%|████▉     | 624/1261 [02:04<02:12,  4.82it/s][A[A
    
     50%|████▉     | 625/1261 [02:05<02:13,  4.76it/s][A[A
    
     50%|████▉     | 626/1261 [02:05<02:13,  4.75it/s][A[A
    
     50%|████▉     | 627/1261 [02:05<02:16,  4.64it/s][A[A
    
     50%|████▉     | 628/1261 [02:05<02:13,  4.76it/s][A[A
    
     50%|████▉     | 629/1261 [02:06<02:14,  4.71it/s][A[A
    
     50%|████▉     | 630/1261 [02:06<02:16,  4.63it/s][A[A
    
     50%|█████     | 631/1261 [02:06<02:16,  4.61it/s][A[A
    
     50%|█████     | 632/1261 [02:06<02:13,  4.72it/s][A[A
    
     50%|█████     | 633/1261 [02:06<02:12,  4.74it/s][A[A
    
     50%|█████     | 634/1261 [02:07<02:10,  4.82it/s][A[A
    
     50%|█████     | 635/1261 [02:07<02:07,  4.89it/s][A[A
    
     50%|█████     | 636/1261 [02:07<02:07,  4.91it/s][A[A
    
     51%|█████     | 637/1261 [02:07<02:05,  4.95it/s][A[A
    
     51%|█████     | 638/1261 [02:07<02:05,  4.95it/s][A[A
    
     51%|█████     | 639/1261 [02:08<02:06,  4.90it/s][A[A
    
     51%|█████     | 640/1261 [02:08<02:05,  4.94it/s][A[A
    
     51%|█████     | 641/1261 [02:08<02:04,  4.96it/s][A[A
    
     51%|█████     | 642/1261 [02:08<02:03,  5.00it/s][A[A
    
     51%|█████     | 643/1261 [02:08<02:01,  5.08it/s][A[A
    
     51%|█████     | 644/1261 [02:09<02:00,  5.13it/s][A[A
    
     51%|█████     | 645/1261 [02:09<01:58,  5.21it/s][A[A
    
     51%|█████     | 646/1261 [02:09<01:56,  5.26it/s][A[A
    
     51%|█████▏    | 647/1261 [02:09<01:55,  5.31it/s][A[A
    
     51%|█████▏    | 648/1261 [02:09<01:56,  5.27it/s][A[A
    
     51%|█████▏    | 649/1261 [02:09<01:55,  5.32it/s][A[A
    
     52%|█████▏    | 650/1261 [02:10<01:54,  5.32it/s][A[A
    
     52%|█████▏    | 651/1261 [02:10<01:54,  5.34it/s][A[A
    
     52%|█████▏    | 652/1261 [02:10<01:53,  5.35it/s][A[A
    
     52%|█████▏    | 653/1261 [02:10<01:53,  5.34it/s][A[A
    
     52%|█████▏    | 654/1261 [02:10<01:54,  5.31it/s][A[A
    
     52%|█████▏    | 655/1261 [02:11<01:54,  5.29it/s][A[A
    
     52%|█████▏    | 656/1261 [02:11<01:55,  5.25it/s][A[A
    
     52%|█████▏    | 657/1261 [02:11<01:54,  5.28it/s][A[A
    
     52%|█████▏    | 658/1261 [02:11<01:54,  5.27it/s][A[A
    
     52%|█████▏    | 659/1261 [02:11<01:54,  5.26it/s][A[A
    
     52%|█████▏    | 660/1261 [02:12<01:53,  5.31it/s][A[A
    
     52%|█████▏    | 661/1261 [02:12<01:52,  5.34it/s][A[A
    
     52%|█████▏    | 662/1261 [02:12<01:52,  5.31it/s][A[A
    
     53%|█████▎    | 663/1261 [02:12<01:53,  5.29it/s][A[A
    
     53%|█████▎    | 664/1261 [02:12<01:57,  5.09it/s][A[A
    
     53%|█████▎    | 665/1261 [02:13<01:59,  5.00it/s][A[A
    
     53%|█████▎    | 666/1261 [02:13<02:01,  4.91it/s][A[A
    
     53%|█████▎    | 667/1261 [02:13<02:02,  4.86it/s][A[A
    
     53%|█████▎    | 668/1261 [02:13<02:01,  4.88it/s][A[A
    
     53%|█████▎    | 669/1261 [02:13<02:00,  4.91it/s][A[A
    
     53%|█████▎    | 670/1261 [02:14<02:00,  4.91it/s][A[A
    
     53%|█████▎    | 671/1261 [02:14<01:59,  4.94it/s][A[A
    
     53%|█████▎    | 672/1261 [02:14<01:59,  4.94it/s][A[A
    
     53%|█████▎    | 673/1261 [02:14<01:58,  4.96it/s][A[A
    
     53%|█████▎    | 674/1261 [02:14<01:58,  4.93it/s][A[A
    
     54%|█████▎    | 675/1261 [02:15<01:58,  4.94it/s][A[A
    
     54%|█████▎    | 676/1261 [02:15<01:58,  4.96it/s][A[A
    
     54%|█████▎    | 677/1261 [02:15<01:57,  4.96it/s][A[A
    
     54%|█████▍    | 678/1261 [02:15<01:57,  4.98it/s][A[A
    
     54%|█████▍    | 679/1261 [02:15<01:59,  4.87it/s][A[A
    
     54%|█████▍    | 680/1261 [02:16<02:01,  4.79it/s][A[A
    
     54%|█████▍    | 681/1261 [02:16<02:01,  4.77it/s][A[A
    
     54%|█████▍    | 682/1261 [02:16<02:02,  4.73it/s][A[A
    
     54%|█████▍    | 683/1261 [02:16<02:00,  4.80it/s][A[A
    
     54%|█████▍    | 684/1261 [02:16<01:59,  4.83it/s][A[A
    
     54%|█████▍    | 685/1261 [02:17<01:58,  4.86it/s][A[A
    
     54%|█████▍    | 686/1261 [02:17<01:57,  4.89it/s][A[A
    
     54%|█████▍    | 687/1261 [02:17<01:54,  5.03it/s][A[A
    
     55%|█████▍    | 688/1261 [02:17<01:53,  5.06it/s][A[A
    
     55%|█████▍    | 689/1261 [02:17<01:58,  4.81it/s][A[A
    
     55%|█████▍    | 690/1261 [02:18<02:00,  4.72it/s][A[A
    
     55%|█████▍    | 691/1261 [02:18<01:57,  4.84it/s][A[A
    
     55%|█████▍    | 692/1261 [02:18<01:59,  4.75it/s][A[A
    
     55%|█████▍    | 693/1261 [02:18<01:56,  4.88it/s][A[A
    
     55%|█████▌    | 694/1261 [02:19<01:56,  4.87it/s][A[A
    
     55%|█████▌    | 695/1261 [02:19<01:56,  4.87it/s][A[A
    
     55%|█████▌    | 696/1261 [02:19<01:53,  4.96it/s][A[A
    
     55%|█████▌    | 697/1261 [02:19<01:53,  4.96it/s][A[A
    
     55%|█████▌    | 698/1261 [02:19<01:56,  4.82it/s][A[A
    
     55%|█████▌    | 699/1261 [02:20<01:55,  4.87it/s][A[A
    
     56%|█████▌    | 700/1261 [02:20<01:55,  4.86it/s][A[A
    
     56%|█████▌    | 701/1261 [02:20<01:56,  4.79it/s][A[A
    
     56%|█████▌    | 702/1261 [02:20<01:53,  4.92it/s][A[A
    
     56%|█████▌    | 703/1261 [02:20<01:50,  5.07it/s][A[A
    
     56%|█████▌    | 704/1261 [02:21<01:47,  5.16it/s][A[A
    
     56%|█████▌    | 705/1261 [02:21<01:46,  5.20it/s][A[A
    
     56%|█████▌    | 706/1261 [02:21<01:49,  5.09it/s][A[A
    
     56%|█████▌    | 707/1261 [02:21<01:50,  5.01it/s][A[A
    
     56%|█████▌    | 708/1261 [02:21<01:52,  4.90it/s][A[A
    
     56%|█████▌    | 709/1261 [02:22<01:55,  4.80it/s][A[A
    
     56%|█████▋    | 710/1261 [02:22<01:53,  4.86it/s][A[A
    
     56%|█████▋    | 711/1261 [02:22<01:54,  4.81it/s][A[A
    
     56%|█████▋    | 712/1261 [02:22<01:54,  4.81it/s][A[A
    
     57%|█████▋    | 713/1261 [02:22<01:50,  4.94it/s][A[A
    
     57%|█████▋    | 714/1261 [02:23<01:50,  4.96it/s][A[A
    
     57%|█████▋    | 715/1261 [02:23<01:48,  5.01it/s][A[A
    
     57%|█████▋    | 716/1261 [02:23<01:47,  5.06it/s][A[A
    
     57%|█████▋    | 717/1261 [02:23<01:48,  5.02it/s][A[A
    
     57%|█████▋    | 718/1261 [02:23<01:45,  5.15it/s][A[A
    
     57%|█████▋    | 719/1261 [02:24<01:47,  5.06it/s][A[A
    
     57%|█████▋    | 720/1261 [02:24<01:45,  5.13it/s][A[A
    
     57%|█████▋    | 721/1261 [02:24<01:45,  5.13it/s][A[A
    
     57%|█████▋    | 722/1261 [02:24<01:46,  5.07it/s][A[A
    
     57%|█████▋    | 723/1261 [02:24<01:44,  5.17it/s][A[A
    
     57%|█████▋    | 724/1261 [02:25<01:45,  5.10it/s][A[A
    
     57%|█████▋    | 725/1261 [02:25<01:43,  5.17it/s][A[A
    
     58%|█████▊    | 726/1261 [02:25<01:45,  5.09it/s][A[A
    
     58%|█████▊    | 727/1261 [02:25<01:42,  5.19it/s][A[A
    
     58%|█████▊    | 728/1261 [02:25<01:41,  5.25it/s][A[A
    
     58%|█████▊    | 729/1261 [02:25<01:40,  5.27it/s][A[A
    
     58%|█████▊    | 730/1261 [02:26<01:41,  5.22it/s][A[A
    
     58%|█████▊    | 731/1261 [02:26<01:42,  5.15it/s][A[A
    
     58%|█████▊    | 732/1261 [02:26<01:41,  5.21it/s][A[A
    
     58%|█████▊    | 733/1261 [02:26<01:40,  5.26it/s][A[A
    
     58%|█████▊    | 734/1261 [02:26<01:40,  5.25it/s][A[A
    
     58%|█████▊    | 735/1261 [02:27<01:40,  5.24it/s][A[A
    
     58%|█████▊    | 736/1261 [02:27<01:40,  5.21it/s][A[A
    
     58%|█████▊    | 737/1261 [02:27<01:40,  5.22it/s][A[A
    
     59%|█████▊    | 738/1261 [02:27<01:41,  5.15it/s][A[A
    
     59%|█████▊    | 739/1261 [02:27<01:40,  5.18it/s][A[A
    
     59%|█████▊    | 740/1261 [02:28<01:43,  5.04it/s][A[A
    
     59%|█████▉    | 741/1261 [02:28<01:45,  4.93it/s][A[A
    
     59%|█████▉    | 742/1261 [02:28<01:46,  4.86it/s][A[A
    
     59%|█████▉    | 743/1261 [02:28<01:46,  4.86it/s][A[A
    
     59%|█████▉    | 744/1261 [02:28<01:46,  4.87it/s][A[A
    
     59%|█████▉    | 745/1261 [02:29<01:45,  4.88it/s][A[A
    
     59%|█████▉    | 746/1261 [02:29<01:48,  4.75it/s][A[A
    
     59%|█████▉    | 747/1261 [02:29<01:50,  4.67it/s][A[A
    
     59%|█████▉    | 748/1261 [02:29<01:49,  4.68it/s][A[A
    
     59%|█████▉    | 749/1261 [02:29<01:45,  4.83it/s][A[A
    
     59%|█████▉    | 750/1261 [02:30<01:47,  4.76it/s][A[A
    
     60%|█████▉    | 751/1261 [02:30<01:43,  4.92it/s][A[A
    
     60%|█████▉    | 752/1261 [02:30<01:46,  4.79it/s][A[A
    
     60%|█████▉    | 753/1261 [02:30<01:43,  4.90it/s][A[A
    
     60%|█████▉    | 754/1261 [02:31<01:44,  4.85it/s][A[A
    
     60%|█████▉    | 755/1261 [02:31<01:46,  4.77it/s][A[A
    
     60%|█████▉    | 756/1261 [02:31<01:42,  4.92it/s][A[A
    
     60%|██████    | 757/1261 [02:31<01:43,  4.88it/s][A[A
    
     60%|██████    | 758/1261 [02:31<01:41,  4.96it/s][A[A
    
     60%|██████    | 759/1261 [02:32<01:40,  5.02it/s][A[A
    
     60%|██████    | 760/1261 [02:32<01:41,  4.96it/s][A[A
    
     60%|██████    | 761/1261 [02:32<01:43,  4.84it/s][A[A
    
     60%|██████    | 762/1261 [02:32<01:44,  4.79it/s][A[A
    
     61%|██████    | 763/1261 [02:32<01:43,  4.83it/s][A[A
    
     61%|██████    | 764/1261 [02:33<01:41,  4.89it/s][A[A
    
     61%|██████    | 765/1261 [02:33<01:41,  4.88it/s][A[A
    
     61%|██████    | 766/1261 [02:33<01:41,  4.90it/s][A[A
    
     61%|██████    | 767/1261 [02:33<01:41,  4.85it/s][A[A
    
     61%|██████    | 768/1261 [02:33<01:42,  4.82it/s][A[A
    
     61%|██████    | 769/1261 [02:34<01:41,  4.84it/s][A[A
    
     61%|██████    | 770/1261 [02:34<01:40,  4.89it/s][A[A
    
     61%|██████    | 771/1261 [02:34<01:40,  4.86it/s][A[A
    
     61%|██████    | 772/1261 [02:34<01:40,  4.87it/s][A[A
    
     61%|██████▏   | 773/1261 [02:34<01:39,  4.90it/s][A[A
    
     61%|██████▏   | 774/1261 [02:35<01:38,  4.95it/s][A[A
    
     61%|██████▏   | 775/1261 [02:35<01:38,  4.92it/s][A[A
    
     62%|██████▏   | 776/1261 [02:35<01:39,  4.85it/s][A[A
    
     62%|██████▏   | 777/1261 [02:35<01:38,  4.89it/s][A[A
    
     62%|██████▏   | 778/1261 [02:35<01:38,  4.90it/s][A[A
    
     62%|██████▏   | 779/1261 [02:36<01:38,  4.92it/s][A[A
    
     62%|██████▏   | 780/1261 [02:36<01:37,  4.91it/s][A[A
    
     62%|██████▏   | 781/1261 [02:36<01:38,  4.88it/s][A[A
    
     62%|██████▏   | 782/1261 [02:36<01:37,  4.91it/s][A[A
    
     62%|██████▏   | 783/1261 [02:36<01:37,  4.91it/s][A[A
    
     62%|██████▏   | 784/1261 [02:37<01:37,  4.91it/s][A[A
    
     62%|██████▏   | 785/1261 [02:37<01:37,  4.91it/s][A[A
    
     62%|██████▏   | 786/1261 [02:37<01:37,  4.86it/s][A[A
    
     62%|██████▏   | 787/1261 [02:37<01:37,  4.84it/s][A[A
    
     62%|██████▏   | 788/1261 [02:37<01:36,  4.88it/s][A[A
    
     63%|██████▎   | 789/1261 [02:38<01:35,  4.92it/s][A[A
    
     63%|██████▎   | 790/1261 [02:38<01:35,  4.93it/s][A[A
    
     63%|██████▎   | 791/1261 [02:38<01:36,  4.85it/s][A[A
    
     63%|██████▎   | 792/1261 [02:38<01:36,  4.86it/s][A[A
    
     63%|██████▎   | 793/1261 [02:38<01:35,  4.91it/s][A[A
    
     63%|██████▎   | 794/1261 [02:39<01:34,  4.93it/s][A[A
    
     63%|██████▎   | 795/1261 [02:39<01:34,  4.95it/s][A[A
    
     63%|██████▎   | 796/1261 [02:39<01:35,  4.87it/s][A[A
    
     63%|██████▎   | 797/1261 [02:39<01:34,  4.89it/s][A[A
    
     63%|██████▎   | 798/1261 [02:40<01:34,  4.88it/s][A[A
    
     63%|██████▎   | 799/1261 [02:40<01:34,  4.88it/s][A[A
    
     63%|██████▎   | 800/1261 [02:40<01:33,  4.94it/s][A[A
    
     64%|██████▎   | 801/1261 [02:40<01:34,  4.88it/s][A[A
    
     64%|██████▎   | 802/1261 [02:40<01:32,  4.98it/s][A[A
    
     64%|██████▎   | 803/1261 [02:41<01:30,  5.04it/s][A[A
    
     64%|██████▍   | 804/1261 [02:41<01:29,  5.13it/s][A[A
    
     64%|██████▍   | 805/1261 [02:41<01:28,  5.17it/s][A[A
    
     64%|██████▍   | 806/1261 [02:41<01:30,  5.05it/s][A[A
    
     64%|██████▍   | 807/1261 [02:41<01:28,  5.11it/s][A[A
    
     64%|██████▍   | 808/1261 [02:41<01:28,  5.14it/s][A[A
    
     64%|██████▍   | 809/1261 [02:42<01:26,  5.21it/s][A[A
    
     64%|██████▍   | 810/1261 [02:42<01:26,  5.19it/s][A[A
    
     64%|██████▍   | 811/1261 [02:42<01:27,  5.17it/s][A[A
    
     64%|██████▍   | 812/1261 [02:42<01:26,  5.20it/s][A[A
    
     64%|██████▍   | 813/1261 [02:42<01:30,  4.96it/s][A[A
    
     65%|██████▍   | 814/1261 [02:43<01:27,  5.09it/s][A[A
    
     65%|██████▍   | 815/1261 [02:43<01:26,  5.18it/s][A[A
    
     65%|██████▍   | 816/1261 [02:43<01:29,  4.98it/s][A[A
    
     65%|██████▍   | 817/1261 [02:43<01:29,  4.97it/s][A[A
    
     65%|██████▍   | 818/1261 [02:43<01:29,  4.93it/s][A[A
    
     65%|██████▍   | 819/1261 [02:44<01:31,  4.86it/s][A[A
    
     65%|██████▌   | 820/1261 [02:44<01:32,  4.76it/s][A[A
    
     65%|██████▌   | 821/1261 [02:44<01:32,  4.75it/s][A[A
    
     65%|██████▌   | 822/1261 [02:44<01:32,  4.77it/s][A[A
    
     65%|██████▌   | 823/1261 [02:45<01:30,  4.82it/s][A[A
    
     65%|██████▌   | 824/1261 [02:45<01:28,  4.95it/s][A[A
    
     65%|██████▌   | 825/1261 [02:45<01:26,  5.05it/s][A[A
    
     66%|██████▌   | 826/1261 [02:45<01:24,  5.12it/s][A[A
    
     66%|██████▌   | 827/1261 [02:45<01:23,  5.19it/s][A[A
    
     66%|██████▌   | 828/1261 [02:45<01:23,  5.21it/s][A[A
    
     66%|██████▌   | 829/1261 [02:46<01:22,  5.22it/s][A[A
    
     66%|██████▌   | 830/1261 [02:46<01:24,  5.09it/s][A[A
    
     66%|██████▌   | 831/1261 [02:46<01:24,  5.10it/s][A[A
    
     66%|██████▌   | 832/1261 [02:46<01:24,  5.09it/s][A[A
    
     66%|██████▌   | 833/1261 [02:46<01:25,  4.98it/s][A[A
    
     66%|██████▌   | 834/1261 [02:47<01:27,  4.89it/s][A[A
    
     66%|██████▌   | 835/1261 [02:47<01:26,  4.93it/s][A[A
    
     66%|██████▋   | 836/1261 [02:47<01:28,  4.80it/s][A[A
    
     66%|██████▋   | 837/1261 [02:47<01:25,  4.94it/s][A[A
    
     66%|██████▋   | 838/1261 [02:47<01:23,  5.08it/s][A[A
    
     67%|██████▋   | 839/1261 [02:48<01:21,  5.16it/s][A[A
    
     67%|██████▋   | 840/1261 [02:48<01:23,  5.02it/s][A[A
    
     67%|██████▋   | 841/1261 [02:48<01:22,  5.12it/s][A[A
    
     67%|██████▋   | 842/1261 [02:48<01:21,  5.16it/s][A[A
    
     67%|██████▋   | 843/1261 [02:48<01:27,  4.80it/s][A[A
    
     67%|██████▋   | 844/1261 [02:49<01:25,  4.86it/s][A[A
    
     67%|██████▋   | 845/1261 [02:49<01:24,  4.90it/s][A[A
    
     67%|██████▋   | 846/1261 [02:49<01:24,  4.93it/s][A[A
    
     67%|██████▋   | 847/1261 [02:49<01:24,  4.92it/s][A[A
    
     67%|██████▋   | 848/1261 [02:50<01:24,  4.90it/s][A[A
    
     67%|██████▋   | 849/1261 [02:50<01:23,  4.94it/s][A[A
    
     67%|██████▋   | 850/1261 [02:50<01:22,  4.96it/s][A[A
    
     67%|██████▋   | 851/1261 [02:50<01:22,  4.97it/s][A[A
    
     68%|██████▊   | 852/1261 [02:50<01:21,  5.00it/s][A[A
    
     68%|██████▊   | 853/1261 [02:50<01:21,  5.02it/s][A[A
    
     68%|██████▊   | 854/1261 [02:51<01:22,  4.96it/s][A[A
    
     68%|██████▊   | 855/1261 [02:51<01:20,  5.05it/s][A[A
    
     68%|██████▊   | 856/1261 [02:51<01:19,  5.09it/s][A[A
    
     68%|██████▊   | 857/1261 [02:51<01:17,  5.18it/s][A[A
    
     68%|██████▊   | 858/1261 [02:51<01:17,  5.22it/s][A[A
    
     68%|██████▊   | 859/1261 [02:52<01:16,  5.26it/s][A[A
    
     68%|██████▊   | 860/1261 [02:52<01:16,  5.28it/s][A[A
    
     68%|██████▊   | 861/1261 [02:52<01:15,  5.31it/s][A[A
    
     68%|██████▊   | 862/1261 [02:52<01:14,  5.35it/s][A[A
    
     68%|██████▊   | 863/1261 [02:52<01:14,  5.35it/s][A[A
    
     69%|██████▊   | 864/1261 [02:53<01:14,  5.32it/s][A[A
    
     69%|██████▊   | 865/1261 [02:53<01:14,  5.35it/s][A[A
    
     69%|██████▊   | 866/1261 [02:53<01:14,  5.29it/s][A[A
    
     69%|██████▉   | 867/1261 [02:53<01:14,  5.28it/s][A[A
    
     69%|██████▉   | 868/1261 [02:53<01:15,  5.24it/s][A[A
    
     69%|██████▉   | 869/1261 [02:54<01:16,  5.11it/s][A[A
    
     69%|██████▉   | 870/1261 [02:54<01:16,  5.12it/s][A[A
    
     69%|██████▉   | 871/1261 [02:54<01:15,  5.14it/s][A[A
    
     69%|██████▉   | 872/1261 [02:54<01:20,  4.86it/s][A[A
    
     69%|██████▉   | 873/1261 [02:54<01:18,  4.97it/s][A[A
    
     69%|██████▉   | 874/1261 [02:55<01:18,  4.93it/s][A[A
    
     69%|██████▉   | 875/1261 [02:55<01:16,  5.03it/s][A[A
    
     69%|██████▉   | 876/1261 [02:55<01:16,  5.01it/s][A[A
    
     70%|██████▉   | 877/1261 [02:55<01:15,  5.09it/s][A[A
    
     70%|██████▉   | 878/1261 [02:55<01:15,  5.06it/s][A[A
    
     70%|██████▉   | 879/1261 [02:56<01:15,  5.06it/s][A[A
    
     70%|██████▉   | 880/1261 [02:56<01:16,  5.00it/s][A[A
    
     70%|██████▉   | 881/1261 [02:56<01:14,  5.13it/s][A[A
    
     70%|██████▉   | 882/1261 [02:56<01:16,  4.95it/s][A[A
    
     70%|███████   | 883/1261 [02:56<01:14,  5.06it/s][A[A
    
     70%|███████   | 884/1261 [02:57<01:16,  4.90it/s][A[A
    
     70%|███████   | 885/1261 [02:57<01:14,  5.05it/s][A[A
    
     70%|███████   | 886/1261 [02:57<01:13,  5.08it/s][A[A
    
     70%|███████   | 887/1261 [02:57<01:14,  5.03it/s][A[A
    
     70%|███████   | 888/1261 [02:57<01:15,  4.94it/s][A[A
    
     70%|███████   | 889/1261 [02:58<01:13,  5.03it/s][A[A
    
     71%|███████   | 890/1261 [02:58<01:12,  5.12it/s][A[A
    
     71%|███████   | 891/1261 [02:58<01:11,  5.19it/s][A[A
    
     71%|███████   | 892/1261 [02:58<01:10,  5.25it/s][A[A
    
     71%|███████   | 893/1261 [02:58<01:09,  5.26it/s][A[A
    
     71%|███████   | 894/1261 [02:58<01:10,  5.24it/s][A[A
    
     71%|███████   | 895/1261 [02:59<01:10,  5.19it/s][A[A
    
     71%|███████   | 896/1261 [02:59<01:11,  5.11it/s][A[A
    
     71%|███████   | 897/1261 [02:59<01:09,  5.21it/s][A[A
    
     71%|███████   | 898/1261 [02:59<01:09,  5.23it/s][A[A
    
     71%|███████▏  | 899/1261 [02:59<01:08,  5.26it/s][A[A
    
     71%|███████▏  | 900/1261 [03:00<01:07,  5.33it/s][A[A
    
     71%|███████▏  | 901/1261 [03:00<01:09,  5.19it/s][A[A
    
     72%|███████▏  | 902/1261 [03:00<01:10,  5.06it/s][A[A
    
     72%|███████▏  | 903/1261 [03:00<01:12,  4.95it/s][A[A
    
     72%|███████▏  | 904/1261 [03:00<01:11,  4.99it/s][A[A
    
     72%|███████▏  | 905/1261 [03:01<01:12,  4.94it/s][A[A
    
     72%|███████▏  | 906/1261 [03:01<01:11,  4.94it/s][A[A
    
     72%|███████▏  | 907/1261 [03:01<01:12,  4.88it/s][A[A
    
     72%|███████▏  | 908/1261 [03:01<01:13,  4.83it/s][A[A
    
     72%|███████▏  | 909/1261 [03:01<01:13,  4.81it/s][A[A
    
     72%|███████▏  | 910/1261 [03:02<01:12,  4.87it/s][A[A
    
     72%|███████▏  | 911/1261 [03:02<01:13,  4.79it/s][A[A
    
     72%|███████▏  | 912/1261 [03:02<01:11,  4.88it/s][A[A
    
     72%|███████▏  | 913/1261 [03:02<01:12,  4.82it/s][A[A
    
     72%|███████▏  | 914/1261 [03:03<01:13,  4.73it/s][A[A
    
     73%|███████▎  | 915/1261 [03:03<01:12,  4.79it/s][A[A
    
     73%|███████▎  | 916/1261 [03:03<01:10,  4.89it/s][A[A
    
     73%|███████▎  | 917/1261 [03:03<01:10,  4.91it/s][A[A
    
     73%|███████▎  | 918/1261 [03:03<01:10,  4.86it/s][A[A
    
     73%|███████▎  | 919/1261 [03:04<01:10,  4.87it/s][A[A
    
     73%|███████▎  | 920/1261 [03:04<01:09,  4.89it/s][A[A
    
     73%|███████▎  | 921/1261 [03:04<01:09,  4.91it/s][A[A
    
     73%|███████▎  | 922/1261 [03:04<01:09,  4.88it/s][A[A
    
     73%|███████▎  | 923/1261 [03:04<01:09,  4.89it/s][A[A
    
     73%|███████▎  | 924/1261 [03:05<01:08,  4.89it/s][A[A
    
     73%|███████▎  | 925/1261 [03:05<01:08,  4.87it/s][A[A
    
     73%|███████▎  | 926/1261 [03:05<01:08,  4.86it/s][A[A
    
     74%|███████▎  | 927/1261 [03:05<01:08,  4.91it/s][A[A
    
     74%|███████▎  | 928/1261 [03:05<01:07,  4.90it/s][A[A
    
     74%|███████▎  | 929/1261 [03:06<01:07,  4.94it/s][A[A
    
     74%|███████▍  | 930/1261 [03:06<01:06,  4.95it/s][A[A
    
     74%|███████▍  | 931/1261 [03:06<01:06,  4.95it/s][A[A
    
     74%|███████▍  | 932/1261 [03:06<01:06,  4.95it/s][A[A
    
     74%|███████▍  | 933/1261 [03:06<01:06,  4.93it/s][A[A
    
     74%|███████▍  | 934/1261 [03:07<01:06,  4.90it/s][A[A
    
     74%|███████▍  | 935/1261 [03:07<01:06,  4.93it/s][A[A
    
     74%|███████▍  | 936/1261 [03:07<01:05,  4.93it/s][A[A
    
     74%|███████▍  | 937/1261 [03:07<01:06,  4.87it/s][A[A
    
     74%|███████▍  | 938/1261 [03:07<01:05,  4.90it/s][A[A
    
     74%|███████▍  | 939/1261 [03:08<01:05,  4.93it/s][A[A
    
     75%|███████▍  | 940/1261 [03:08<01:04,  4.96it/s][A[A
    
     75%|███████▍  | 941/1261 [03:08<01:04,  4.95it/s][A[A
    
     75%|███████▍  | 942/1261 [03:08<01:05,  4.89it/s][A[A
    
     75%|███████▍  | 943/1261 [03:08<01:05,  4.88it/s][A[A
    
     75%|███████▍  | 944/1261 [03:09<01:04,  4.92it/s][A[A
    
     75%|███████▍  | 945/1261 [03:09<01:04,  4.91it/s][A[A
    
     75%|███████▌  | 946/1261 [03:09<01:04,  4.91it/s][A[A
    
     75%|███████▌  | 947/1261 [03:09<01:03,  4.92it/s][A[A
    
     75%|███████▌  | 948/1261 [03:09<01:03,  4.94it/s][A[A
    
     75%|███████▌  | 949/1261 [03:10<01:02,  4.97it/s][A[A
    
     75%|███████▌  | 950/1261 [03:10<01:02,  4.97it/s][A[A
    
     75%|███████▌  | 951/1261 [03:10<01:02,  4.97it/s][A[A
    
     75%|███████▌  | 952/1261 [03:10<01:02,  4.95it/s][A[A
    
     76%|███████▌  | 953/1261 [03:10<01:02,  4.95it/s][A[A
    
     76%|███████▌  | 954/1261 [03:11<01:02,  4.90it/s][A[A
    
     76%|███████▌  | 955/1261 [03:11<01:02,  4.86it/s][A[A
    
     76%|███████▌  | 956/1261 [03:11<01:02,  4.86it/s][A[A
    
     76%|███████▌  | 957/1261 [03:11<01:02,  4.86it/s][A[A
    
     76%|███████▌  | 958/1261 [03:11<01:01,  4.89it/s][A[A
    
     76%|███████▌  | 959/1261 [03:12<01:01,  4.90it/s][A[A
    
     76%|███████▌  | 960/1261 [03:12<01:01,  4.88it/s][A[A
    
     76%|███████▌  | 961/1261 [03:12<01:01,  4.86it/s][A[A
    
     76%|███████▋  | 962/1261 [03:12<01:01,  4.90it/s][A[A
    
     76%|███████▋  | 963/1261 [03:13<01:01,  4.87it/s][A[A
    
     76%|███████▋  | 964/1261 [03:13<01:00,  4.90it/s][A[A
    
     77%|███████▋  | 965/1261 [03:13<01:00,  4.92it/s][A[A
    
     77%|███████▋  | 966/1261 [03:13<00:59,  4.92it/s][A[A
    
     77%|███████▋  | 967/1261 [03:13<00:59,  4.94it/s][A[A
    
     77%|███████▋  | 968/1261 [03:14<00:59,  4.91it/s][A[A
    
     77%|███████▋  | 969/1261 [03:14<00:58,  4.96it/s][A[A
    
     77%|███████▋  | 970/1261 [03:14<00:58,  4.96it/s][A[A
    
     77%|███████▋  | 971/1261 [03:14<00:58,  4.94it/s][A[A
    
     77%|███████▋  | 972/1261 [03:14<00:58,  4.92it/s][A[A
    
     77%|███████▋  | 973/1261 [03:15<00:58,  4.93it/s][A[A
    
     77%|███████▋  | 974/1261 [03:15<00:58,  4.89it/s][A[A
    
     77%|███████▋  | 975/1261 [03:15<00:58,  4.89it/s][A[A
    
     77%|███████▋  | 976/1261 [03:15<00:58,  4.89it/s][A[A
    
     77%|███████▋  | 977/1261 [03:15<00:58,  4.83it/s][A[A
    
     78%|███████▊  | 978/1261 [03:16<00:59,  4.79it/s][A[A
    
     78%|███████▊  | 979/1261 [03:16<00:58,  4.82it/s][A[A
    
     78%|███████▊  | 980/1261 [03:16<00:58,  4.82it/s][A[A
    
     78%|███████▊  | 981/1261 [03:16<00:57,  4.88it/s][A[A
    
     78%|███████▊  | 982/1261 [03:16<00:56,  4.91it/s][A[A
    
     78%|███████▊  | 983/1261 [03:17<00:56,  4.91it/s][A[A
    
     78%|███████▊  | 984/1261 [03:17<00:56,  4.92it/s][A[A
    
     78%|███████▊  | 985/1261 [03:17<00:56,  4.91it/s][A[A
    
     78%|███████▊  | 986/1261 [03:17<00:55,  4.92it/s][A[A
    
     78%|███████▊  | 987/1261 [03:17<00:55,  4.93it/s][A[A
    
     78%|███████▊  | 988/1261 [03:18<00:55,  4.88it/s][A[A
    
     78%|███████▊  | 989/1261 [03:18<00:55,  4.89it/s][A[A
    
     79%|███████▊  | 990/1261 [03:18<00:55,  4.90it/s][A[A
    
     79%|███████▊  | 991/1261 [03:18<00:55,  4.90it/s][A[A
    
     79%|███████▊  | 992/1261 [03:18<00:55,  4.87it/s][A[A
    
     79%|███████▊  | 993/1261 [03:19<00:54,  4.90it/s][A[A
    
     79%|███████▉  | 994/1261 [03:19<00:54,  4.92it/s][A[A
    
     79%|███████▉  | 995/1261 [03:19<00:54,  4.92it/s][A[A
    
     79%|███████▉  | 996/1261 [03:19<00:54,  4.85it/s][A[A
    
     79%|███████▉  | 997/1261 [03:19<00:55,  4.76it/s][A[A
    
     79%|███████▉  | 998/1261 [03:20<00:53,  4.94it/s][A[A
    
     79%|███████▉  | 999/1261 [03:20<00:52,  4.99it/s][A[A
    
     79%|███████▉  | 1000/1261 [03:20<00:51,  5.10it/s][A[A
    
     79%|███████▉  | 1001/1261 [03:20<00:51,  5.01it/s][A[A
    
     79%|███████▉  | 1002/1261 [03:20<00:51,  4.99it/s][A[A
    
     80%|███████▉  | 1003/1261 [03:21<00:50,  5.14it/s][A[A
    
     80%|███████▉  | 1004/1261 [03:21<00:49,  5.22it/s][A[A
    
     80%|███████▉  | 1005/1261 [03:21<00:48,  5.27it/s][A[A
    
     80%|███████▉  | 1006/1261 [03:21<00:48,  5.30it/s][A[A
    
     80%|███████▉  | 1007/1261 [03:21<00:47,  5.32it/s][A[A
    
     80%|███████▉  | 1008/1261 [03:22<00:47,  5.30it/s][A[A
    
     80%|████████  | 1009/1261 [03:22<00:47,  5.32it/s][A[A
    
     80%|████████  | 1010/1261 [03:22<00:47,  5.28it/s][A[A
    
     80%|████████  | 1011/1261 [03:22<00:46,  5.32it/s][A[A
    
     80%|████████  | 1012/1261 [03:22<00:46,  5.34it/s][A[A
    
     80%|████████  | 1013/1261 [03:22<00:46,  5.33it/s][A[A
    
     80%|████████  | 1014/1261 [03:23<00:46,  5.35it/s][A[A
    
     80%|████████  | 1015/1261 [03:23<00:45,  5.39it/s][A[A
    
     81%|████████  | 1016/1261 [03:23<00:46,  5.22it/s][A[A
    
     81%|████████  | 1017/1261 [03:23<00:46,  5.28it/s][A[A
    
     81%|████████  | 1018/1261 [03:23<00:45,  5.30it/s][A[A
    
     81%|████████  | 1019/1261 [03:24<00:45,  5.34it/s][A[A
    
     81%|████████  | 1020/1261 [03:24<00:46,  5.17it/s][A[A
    
     81%|████████  | 1021/1261 [03:24<00:46,  5.21it/s][A[A
    
     81%|████████  | 1022/1261 [03:24<00:46,  5.15it/s][A[A
    
     81%|████████  | 1023/1261 [03:24<00:48,  4.95it/s][A[A
    
     81%|████████  | 1024/1261 [03:25<00:48,  4.93it/s][A[A
    
     81%|████████▏ | 1025/1261 [03:25<00:47,  5.01it/s][A[A
    
     81%|████████▏ | 1026/1261 [03:25<00:47,  4.90it/s][A[A
    
     81%|████████▏ | 1027/1261 [03:25<00:47,  4.88it/s][A[A
    
     82%|████████▏ | 1028/1261 [03:25<00:47,  4.89it/s][A[A
    
     82%|████████▏ | 1029/1261 [03:26<00:46,  4.96it/s][A[A
    
     82%|████████▏ | 1030/1261 [03:26<00:47,  4.89it/s][A[A
    
     82%|████████▏ | 1031/1261 [03:26<00:47,  4.82it/s][A[A
    
     82%|████████▏ | 1032/1261 [03:26<00:48,  4.67it/s][A[A
    
     82%|████████▏ | 1033/1261 [03:27<00:48,  4.72it/s][A[A
    
     82%|████████▏ | 1034/1261 [03:27<00:46,  4.90it/s][A[A
    
     82%|████████▏ | 1035/1261 [03:27<00:45,  4.98it/s][A[A
    
     82%|████████▏ | 1036/1261 [03:27<00:44,  5.06it/s][A[A
    
     82%|████████▏ | 1037/1261 [03:27<00:43,  5.12it/s][A[A
    
     82%|████████▏ | 1038/1261 [03:27<00:42,  5.19it/s][A[A
    
     82%|████████▏ | 1039/1261 [03:28<00:42,  5.22it/s][A[A
    
     82%|████████▏ | 1040/1261 [03:28<00:42,  5.20it/s][A[A
    
     83%|████████▎ | 1041/1261 [03:28<00:41,  5.26it/s][A[A
    
     83%|████████▎ | 1042/1261 [03:28<00:41,  5.29it/s][A[A
    
     83%|████████▎ | 1043/1261 [03:28<00:40,  5.32it/s][A[A
    
     83%|████████▎ | 1044/1261 [03:29<00:40,  5.32it/s][A[A
    
     83%|████████▎ | 1045/1261 [03:29<00:40,  5.36it/s][A[A
    
     83%|████████▎ | 1046/1261 [03:29<00:40,  5.37it/s][A[A
    
     83%|████████▎ | 1047/1261 [03:29<00:39,  5.38it/s][A[A
    
     83%|████████▎ | 1048/1261 [03:29<00:39,  5.37it/s][A[A
    
     83%|████████▎ | 1049/1261 [03:30<00:39,  5.38it/s][A[A
    
     83%|████████▎ | 1050/1261 [03:30<00:39,  5.35it/s][A[A
    
     83%|████████▎ | 1051/1261 [03:30<00:39,  5.29it/s][A[A
    
     83%|████████▎ | 1052/1261 [03:30<00:39,  5.33it/s][A[A
    
     84%|████████▎ | 1053/1261 [03:30<00:38,  5.35it/s][A[A
    
     84%|████████▎ | 1054/1261 [03:30<00:38,  5.32it/s][A[A
    
     84%|████████▎ | 1055/1261 [03:31<00:38,  5.34it/s][A[A
    
     84%|████████▎ | 1056/1261 [03:31<00:38,  5.27it/s][A[A
    
     84%|████████▍ | 1057/1261 [03:31<00:38,  5.29it/s][A[A
    
     84%|████████▍ | 1058/1261 [03:31<00:38,  5.31it/s][A[A
    
     84%|████████▍ | 1059/1261 [03:31<00:38,  5.24it/s][A[A
    
     84%|████████▍ | 1060/1261 [03:32<00:38,  5.20it/s][A[A
    
     84%|████████▍ | 1061/1261 [03:32<00:39,  5.13it/s][A[A
    
     84%|████████▍ | 1062/1261 [03:32<00:39,  5.04it/s][A[A
    
     84%|████████▍ | 1063/1261 [03:32<00:39,  5.01it/s][A[A
    
     84%|████████▍ | 1064/1261 [03:32<00:39,  4.96it/s][A[A
    
     84%|████████▍ | 1065/1261 [03:33<00:39,  4.96it/s][A[A
    
     85%|████████▍ | 1066/1261 [03:33<00:39,  4.92it/s][A[A
    
     85%|████████▍ | 1067/1261 [03:33<00:39,  4.86it/s][A[A
    
     85%|████████▍ | 1068/1261 [03:33<00:39,  4.86it/s][A[A
    
     85%|████████▍ | 1069/1261 [03:33<00:39,  4.86it/s][A[A
    
     85%|████████▍ | 1070/1261 [03:34<00:39,  4.87it/s][A[A
    
     85%|████████▍ | 1071/1261 [03:34<00:38,  4.91it/s][A[A
    
     85%|████████▌ | 1072/1261 [03:34<00:38,  4.92it/s][A[A
    
     85%|████████▌ | 1073/1261 [03:34<00:38,  4.94it/s][A[A
    
     85%|████████▌ | 1074/1261 [03:34<00:37,  4.94it/s][A[A
    
     85%|████████▌ | 1075/1261 [03:35<00:37,  4.96it/s][A[A
    
     85%|████████▌ | 1076/1261 [03:35<00:37,  4.94it/s][A[A
    
     85%|████████▌ | 1077/1261 [03:35<00:37,  4.89it/s][A[A
    
     85%|████████▌ | 1078/1261 [03:35<00:37,  4.87it/s][A[A
    
     86%|████████▌ | 1079/1261 [03:35<00:37,  4.90it/s][A[A
    
     86%|████████▌ | 1080/1261 [03:36<00:35,  5.05it/s][A[A
    
     86%|████████▌ | 1081/1261 [03:36<00:35,  5.14it/s][A[A
    
     86%|████████▌ | 1082/1261 [03:36<00:34,  5.20it/s][A[A
    
     86%|████████▌ | 1083/1261 [03:36<00:34,  5.18it/s][A[A
    
     86%|████████▌ | 1084/1261 [03:36<00:34,  5.06it/s][A[A
    
     86%|████████▌ | 1085/1261 [03:37<00:35,  4.99it/s][A[A
    
     86%|████████▌ | 1086/1261 [03:37<00:34,  5.01it/s][A[A
    
     86%|████████▌ | 1087/1261 [03:37<00:34,  4.99it/s][A[A
    
     86%|████████▋ | 1088/1261 [03:37<00:35,  4.85it/s][A[A
    
     86%|████████▋ | 1089/1261 [03:37<00:35,  4.83it/s][A[A
    
     86%|████████▋ | 1090/1261 [03:38<00:35,  4.86it/s][A[A
    
     87%|████████▋ | 1091/1261 [03:38<00:35,  4.79it/s][A[A
    
     87%|████████▋ | 1092/1261 [03:38<00:35,  4.74it/s][A[A
    
     87%|████████▋ | 1093/1261 [03:38<00:36,  4.66it/s][A[A
    
     87%|████████▋ | 1094/1261 [03:39<00:34,  4.78it/s][A[A
    
     87%|████████▋ | 1095/1261 [03:39<00:35,  4.71it/s][A[A
    
     87%|████████▋ | 1096/1261 [03:39<00:35,  4.70it/s][A[A
    
     87%|████████▋ | 1097/1261 [03:39<00:35,  4.68it/s][A[A
    
     87%|████████▋ | 1098/1261 [03:39<00:35,  4.62it/s][A[A
    
     87%|████████▋ | 1099/1261 [03:40<00:34,  4.64it/s][A[A
    
     87%|████████▋ | 1100/1261 [03:40<00:34,  4.68it/s][A[A
    
     87%|████████▋ | 1101/1261 [03:40<00:34,  4.66it/s][A[A
    
     87%|████████▋ | 1102/1261 [03:40<00:33,  4.77it/s][A[A
    
     87%|████████▋ | 1103/1261 [03:40<00:33,  4.69it/s][A[A
    
     88%|████████▊ | 1104/1261 [03:41<00:32,  4.87it/s][A[A
    
     88%|████████▊ | 1105/1261 [03:41<00:31,  4.95it/s][A[A
    
     88%|████████▊ | 1106/1261 [03:41<00:30,  5.07it/s][A[A
    
     88%|████████▊ | 1107/1261 [03:41<00:29,  5.17it/s][A[A
    
     88%|████████▊ | 1108/1261 [03:41<00:29,  5.20it/s][A[A
    
     88%|████████▊ | 1109/1261 [03:42<00:29,  5.21it/s][A[A
    
     88%|████████▊ | 1110/1261 [03:42<00:28,  5.29it/s][A[A
    
     88%|████████▊ | 1111/1261 [03:42<00:28,  5.26it/s][A[A
    
     88%|████████▊ | 1112/1261 [03:42<00:28,  5.27it/s][A[A
    
     88%|████████▊ | 1113/1261 [03:42<00:29,  5.07it/s][A[A
    
     88%|████████▊ | 1114/1261 [03:43<00:29,  4.99it/s][A[A
    
     88%|████████▊ | 1115/1261 [03:43<00:28,  5.07it/s][A[A
    
     89%|████████▊ | 1116/1261 [03:43<00:28,  5.12it/s][A[A
    
     89%|████████▊ | 1117/1261 [03:43<00:28,  5.12it/s][A[A
    
     89%|████████▊ | 1118/1261 [03:43<00:28,  5.08it/s][A[A
    
     89%|████████▊ | 1119/1261 [03:44<00:28,  5.05it/s][A[A
    
     89%|████████▉ | 1120/1261 [03:44<00:27,  5.15it/s][A[A
    
     89%|████████▉ | 1121/1261 [03:44<00:27,  5.17it/s][A[A
    
     89%|████████▉ | 1122/1261 [03:44<00:26,  5.20it/s][A[A
    
     89%|████████▉ | 1123/1261 [03:44<00:26,  5.22it/s][A[A
    
     89%|████████▉ | 1124/1261 [03:45<00:26,  5.16it/s][A[A
    
     89%|████████▉ | 1125/1261 [03:45<00:26,  5.21it/s][A[A
    
     89%|████████▉ | 1126/1261 [03:45<00:25,  5.23it/s][A[A
    
     89%|████████▉ | 1127/1261 [03:45<00:25,  5.25it/s][A[A
    
     89%|████████▉ | 1128/1261 [03:45<00:25,  5.27it/s][A[A
    
     90%|████████▉ | 1129/1261 [03:45<00:25,  5.18it/s][A[A
    
     90%|████████▉ | 1130/1261 [03:46<00:25,  5.23it/s][A[A
    
     90%|████████▉ | 1131/1261 [03:46<00:24,  5.29it/s][A[A
    
     90%|████████▉ | 1132/1261 [03:46<00:24,  5.16it/s][A[A
    
     90%|████████▉ | 1133/1261 [03:46<00:24,  5.20it/s][A[A
    
     90%|████████▉ | 1134/1261 [03:46<00:24,  5.19it/s][A[A
    
     90%|█████████ | 1135/1261 [03:47<00:24,  5.18it/s][A[A
    
     90%|█████████ | 1136/1261 [03:47<00:23,  5.21it/s][A[A
    
     90%|█████████ | 1137/1261 [03:47<00:23,  5.23it/s][A[A
    
     90%|█████████ | 1138/1261 [03:47<00:23,  5.15it/s][A[A
    
     90%|█████████ | 1139/1261 [03:47<00:23,  5.15it/s][A[A
    
     90%|█████████ | 1140/1261 [03:48<00:23,  5.17it/s][A[A
    
     90%|█████████ | 1141/1261 [03:48<00:23,  5.16it/s][A[A
    
     91%|█████████ | 1142/1261 [03:48<00:23,  5.17it/s][A[A
    
     91%|█████████ | 1143/1261 [03:48<00:22,  5.14it/s][A[A
    
     91%|█████████ | 1144/1261 [03:48<00:22,  5.11it/s][A[A
    
     91%|█████████ | 1145/1261 [03:49<00:22,  5.08it/s][A[A
    
     91%|█████████ | 1146/1261 [03:49<00:22,  5.05it/s][A[A
    
     91%|█████████ | 1147/1261 [03:49<00:22,  5.00it/s][A[A
    
     91%|█████████ | 1148/1261 [03:49<00:22,  4.98it/s][A[A
    
     91%|█████████ | 1149/1261 [03:49<00:22,  4.99it/s][A[A
    
     91%|█████████ | 1150/1261 [03:50<00:22,  4.96it/s][A[A
    
     91%|█████████▏| 1151/1261 [03:50<00:22,  4.88it/s][A[A
    
     91%|█████████▏| 1152/1261 [03:50<00:22,  4.84it/s][A[A
    
     91%|█████████▏| 1153/1261 [03:50<00:23,  4.67it/s][A[A
    
     92%|█████████▏| 1154/1261 [03:50<00:22,  4.76it/s][A[A
    
     92%|█████████▏| 1155/1261 [03:51<00:22,  4.73it/s][A[A
    
     92%|█████████▏| 1156/1261 [03:51<00:21,  4.81it/s][A[A
    
     92%|█████████▏| 1157/1261 [03:51<00:20,  4.97it/s][A[A
    
     92%|█████████▏| 1158/1261 [03:51<00:20,  4.92it/s][A[A
    
     92%|█████████▏| 1159/1261 [03:51<00:20,  5.06it/s][A[A
    
     92%|█████████▏| 1160/1261 [03:52<00:19,  5.09it/s][A[A
    
     92%|█████████▏| 1161/1261 [03:52<00:19,  5.13it/s][A[A
    
     92%|█████████▏| 1162/1261 [03:52<00:19,  5.13it/s][A[A
    
     92%|█████████▏| 1163/1261 [03:52<00:19,  5.12it/s][A[A
    
     92%|█████████▏| 1164/1261 [03:52<00:18,  5.12it/s][A[A
    
     92%|█████████▏| 1165/1261 [03:53<00:18,  5.19it/s][A[A
    
     92%|█████████▏| 1166/1261 [03:53<00:18,  5.22it/s][A[A
    
     93%|█████████▎| 1167/1261 [03:53<00:18,  5.20it/s][A[A
    
     93%|█████████▎| 1168/1261 [03:53<00:17,  5.21it/s][A[A
    
     93%|█████████▎| 1169/1261 [03:53<00:17,  5.18it/s][A[A
    
     93%|█████████▎| 1170/1261 [03:54<00:17,  5.17it/s][A[A
    
     93%|█████████▎| 1171/1261 [03:54<00:17,  5.10it/s][A[A
    
     93%|█████████▎| 1172/1261 [03:54<00:17,  5.04it/s][A[A
    
     93%|█████████▎| 1173/1261 [03:54<00:17,  5.13it/s][A[A
    
     93%|█████████▎| 1174/1261 [03:54<00:17,  5.11it/s][A[A
    
     93%|█████████▎| 1175/1261 [03:55<00:17,  5.01it/s][A[A
    
     93%|█████████▎| 1176/1261 [03:55<00:17,  4.89it/s][A[A
    
     93%|█████████▎| 1177/1261 [03:55<00:17,  4.88it/s][A[A
    
     93%|█████████▎| 1178/1261 [03:55<00:16,  5.02it/s][A[A
    
     93%|█████████▎| 1179/1261 [03:55<00:16,  5.07it/s][A[A
    
     94%|█████████▎| 1180/1261 [03:56<00:16,  5.03it/s][A[A
    
     94%|█████████▎| 1181/1261 [03:56<00:15,  5.11it/s][A[A
    
     94%|█████████▎| 1182/1261 [03:56<00:15,  4.98it/s][A[A
    
     94%|█████████▍| 1183/1261 [03:56<00:15,  5.10it/s][A[A
    
     94%|█████████▍| 1184/1261 [03:56<00:14,  5.15it/s][A[A
    
     94%|█████████▍| 1185/1261 [03:57<00:15,  4.97it/s][A[A
    
     94%|█████████▍| 1186/1261 [03:57<00:14,  5.05it/s][A[A
    
     94%|█████████▍| 1187/1261 [03:57<00:14,  5.08it/s][A[A
    
     94%|█████████▍| 1188/1261 [03:57<00:14,  5.13it/s][A[A
    
     94%|█████████▍| 1189/1261 [03:57<00:13,  5.15it/s][A[A
    
     94%|█████████▍| 1190/1261 [03:58<00:13,  5.10it/s][A[A
    
     94%|█████████▍| 1191/1261 [03:58<00:13,  5.08it/s][A[A
    
     95%|█████████▍| 1192/1261 [03:58<00:13,  5.10it/s][A[A
    
     95%|█████████▍| 1193/1261 [03:58<00:13,  5.06it/s][A[A
    
     95%|█████████▍| 1194/1261 [03:58<00:13,  5.13it/s][A[A
    
     95%|█████████▍| 1195/1261 [03:59<00:12,  5.10it/s][A[A
    
     95%|█████████▍| 1196/1261 [03:59<00:13,  4.91it/s][A[A
    
     95%|█████████▍| 1197/1261 [03:59<00:12,  4.97it/s][A[A
    
     95%|█████████▌| 1198/1261 [03:59<00:12,  4.90it/s][A[A
    
     95%|█████████▌| 1199/1261 [03:59<00:12,  4.83it/s][A[A
    
     95%|█████████▌| 1200/1261 [04:00<00:12,  4.80it/s][A[A
    
     95%|█████████▌| 1201/1261 [04:00<00:12,  4.76it/s][A[A
    
     95%|█████████▌| 1202/1261 [04:00<00:12,  4.78it/s][A[A
    
     95%|█████████▌| 1203/1261 [04:00<00:12,  4.80it/s][A[A
    
     95%|█████████▌| 1204/1261 [04:00<00:11,  4.93it/s][A[A
    
     96%|█████████▌| 1205/1261 [04:01<00:11,  4.98it/s][A[A
    
     96%|█████████▌| 1206/1261 [04:01<00:10,  5.08it/s][A[A
    
     96%|█████████▌| 1207/1261 [04:01<00:10,  4.93it/s][A[A
    
     96%|█████████▌| 1208/1261 [04:01<00:10,  5.04it/s][A[A
    
     96%|█████████▌| 1209/1261 [04:01<00:10,  4.90it/s][A[A
    
     96%|█████████▌| 1210/1261 [04:02<00:10,  5.00it/s][A[A
    
     96%|█████████▌| 1211/1261 [04:02<00:10,  4.85it/s][A[A
    
     96%|█████████▌| 1212/1261 [04:02<00:10,  4.90it/s][A[A
    
     96%|█████████▌| 1213/1261 [04:02<00:09,  4.93it/s][A[A
    
     96%|█████████▋| 1214/1261 [04:02<00:09,  4.92it/s][A[A
    
     96%|█████████▋| 1215/1261 [04:03<00:09,  4.88it/s][A[A
    
     96%|█████████▋| 1216/1261 [04:03<00:09,  4.92it/s][A[A
    
     97%|█████████▋| 1217/1261 [04:03<00:08,  4.94it/s][A[A
    
     97%|█████████▋| 1218/1261 [04:03<00:08,  4.93it/s][A[A
    
     97%|█████████▋| 1219/1261 [04:03<00:08,  4.94it/s][A[A
    
     97%|█████████▋| 1220/1261 [04:04<00:08,  4.91it/s][A[A
    
     97%|█████████▋| 1221/1261 [04:04<00:08,  4.85it/s][A[A
    
     97%|█████████▋| 1222/1261 [04:04<00:08,  4.73it/s][A[A
    
     97%|█████████▋| 1223/1261 [04:04<00:08,  4.74it/s][A[A
    
     97%|█████████▋| 1224/1261 [04:04<00:07,  4.73it/s][A[A
    
     97%|█████████▋| 1225/1261 [04:05<00:07,  4.70it/s][A[A
    
     97%|█████████▋| 1226/1261 [04:05<00:07,  4.73it/s][A[A
    
     97%|█████████▋| 1227/1261 [04:05<00:07,  4.78it/s][A[A
    
     97%|█████████▋| 1228/1261 [04:05<00:06,  4.81it/s][A[A
    
     97%|█████████▋| 1229/1261 [04:06<00:06,  4.78it/s][A[A
    
     98%|█████████▊| 1230/1261 [04:06<00:06,  4.85it/s][A[A
    
     98%|█████████▊| 1231/1261 [04:06<00:06,  4.84it/s][A[A
    
     98%|█████████▊| 1232/1261 [04:06<00:06,  4.80it/s][A[A
    
     98%|█████████▊| 1233/1261 [04:06<00:05,  4.85it/s][A[A
    
     98%|█████████▊| 1234/1261 [04:07<00:05,  4.86it/s][A[A
    
     98%|█████████▊| 1235/1261 [04:07<00:05,  4.81it/s][A[A
    
     98%|█████████▊| 1236/1261 [04:07<00:05,  4.86it/s][A[A
    
     98%|█████████▊| 1237/1261 [04:07<00:04,  4.81it/s][A[A
    
     98%|█████████▊| 1238/1261 [04:07<00:04,  4.82it/s][A[A
    
     98%|█████████▊| 1239/1261 [04:08<00:04,  4.79it/s][A[A
    
     98%|█████████▊| 1240/1261 [04:08<00:04,  4.83it/s][A[A
    
     98%|█████████▊| 1241/1261 [04:08<00:04,  4.76it/s][A[A
    
     98%|█████████▊| 1242/1261 [04:08<00:03,  4.80it/s][A[A
    
     99%|█████████▊| 1243/1261 [04:08<00:03,  4.75it/s][A[A
    
     99%|█████████▊| 1244/1261 [04:09<00:03,  4.83it/s][A[A
    
     99%|█████████▊| 1245/1261 [04:09<00:03,  4.92it/s][A[A
    
     99%|█████████▉| 1246/1261 [04:09<00:03,  4.64it/s][A[A
    
     99%|█████████▉| 1247/1261 [04:09<00:02,  4.84it/s][A[A
    
     99%|█████████▉| 1248/1261 [04:09<00:02,  4.90it/s][A[A
    
     99%|█████████▉| 1249/1261 [04:10<00:02,  5.03it/s][A[A
    
     99%|█████████▉| 1250/1261 [04:10<00:02,  5.08it/s][A[A
    
     99%|█████████▉| 1251/1261 [04:10<00:01,  5.16it/s][A[A
    
     99%|█████████▉| 1252/1261 [04:10<00:01,  5.25it/s][A[A
    
     99%|█████████▉| 1253/1261 [04:10<00:01,  5.27it/s][A[A
    
     99%|█████████▉| 1254/1261 [04:11<00:01,  5.32it/s][A[A
    
    100%|█████████▉| 1255/1261 [04:11<00:01,  5.27it/s][A[A
    
    100%|█████████▉| 1256/1261 [04:11<00:00,  5.24it/s][A[A
    
    100%|█████████▉| 1257/1261 [04:11<00:00,  5.06it/s][A[A
    
    100%|█████████▉| 1258/1261 [04:11<00:00,  4.88it/s][A[A
    
    100%|█████████▉| 1259/1261 [04:12<00:00,  5.04it/s][A[A
    
    100%|█████████▉| 1260/1261 [04:12<00:00,  5.14it/s][A[A
    
    [A[A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: lane_lines_output.mp4 
    
    CPU times: user 5min 36s, sys: 9.92 s, total: 5min 46s
    Wall time: 4min 13s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```





<video width="960" height="540" controls>
  <source src="lane_lines_output.mp4">
</video>




## Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

**What issues would come up**

- Shadows would severely affect how the image would be perceived by the various algorithms. The following algorithms should be somewhat stable in relatively bright lighting but if you were to have darker photos or a shaky camera in the video, it would be make all of the above algorithms kind of obsolete.
- The thresholds being chosen for this project may work for this but if it may not work in all situations/countries. In america, there is only one solid color line either the left or right on the lane if the vehicle was driving in the most left/right lanes but in countries like Singapore where there are 2 solid color lines, it would might causes. This needs to be tried out.

**How to improve the project further**

- Functions shouldn't be done in the ipython notebook but instead should be in seperate files. The output of those functions has to be small and more lightweight. Some of the functions written in this notebook was inspired by the udacity notes and the code is too tied together between functions - especially the section for identifying the lane pixels
- Functions are not encapsulated but instead, they keep picking up the details from the main environment which is kind of bad practise. This will be further improved in future iterations.
- Test against different conditions where roads are darker in color etc. However, rather than continuing this approach, it might be better to see if a deep learning technique to do semantic segmentation would be helpful in this scenario.



```python

```
