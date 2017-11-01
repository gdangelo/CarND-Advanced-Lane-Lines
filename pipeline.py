import os
import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Parameters for calibration
mtx = []
dist = []
calibration_file = "calibration_pickle.p"

# Parameters for gradient threshold
ksize = 7
gradx_thresh = (25, 255)
grady_thresh = (25, 255)
magni_thresh = (25, 255)
dir_thresh = (0., 0.09)
hls_thresh = (125, 255)

def get_points_for_calibration(nx, ny):
    # Prepare object points
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    for idx, fname in enumerate(images):
        # Read image
        img = cv2.imread(fname)

        # Convert image in grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners (for an 9x6 board)
        ret, corners = cv2.findChessboardCorners(img, (nx,ny), None)

        if (ret == True):
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imwrite('./output_images/calibration/corners_found' + str(idx) + '.jpg', img)

    return (objpoints, imgpoints)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = 1 if orient == 'x' else 0
    y = 1 if orient == 'y' else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    sobel_scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    grad_binary = np.zeros_like(sobel_scaled)
    grad_binary[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(np.add(np.square(sobelx), np.square(sobely)))
    sobel_scaled = np.uint8(255*mag/np.max(mag))
    # Apply threshold
    mag_binary = np.zeros_like(sobel_scaled)
    mag_binary[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # Apply threshold
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return dir_binary

def hls_threshold(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S_channel = hls[:,:,2]
    color_binary = np.zeros_like(S_channel)
    color_binary[(S_channel >= thresh[0]) & (S_channel <= thresh[1])] = 1
    return color_binary

def color_gradient_threshold(img):
    # Apply color gradient (S channel)
    hls_binary = hls_threshold(img, thresh=hls_thresh)

    # Apply gradient thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=gradx_thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=grady_thresh)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=magni_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)

    # Combine gradient & color thresholds
    color_binary = np.dstack((np.zeros_like(dir_binary), dir_binary, dir_binary)) * 255
    gradient_binary = np.zeros_like(dir_binary)
    gradient_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined = np.zeros_like(dir_binary)
    combined[(gradient_binary == 1) | (hls_binary == 1)] = 1

    return combined

def perspective_transform(img):
    # From trapezoidale shape on straight lines...
    src = np.float32([[519, 502], [765, 502], [1029, 668], [275, 668]])
    # ...to rectangle
    dst = np.float32([[275, 502], [1029, 502], [1029, 668], [275, 668]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    return (cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR), Minv)

def sliding_windows_polyfit(img, previous_left_fit=None, previous_right_fit=None):

    # Get indices of all nonzero pixels along x and y axis
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    # Set margin for searching
    margin = 100
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Look for lines from scratch ('blind search')
    if (previous_left_fit is None or previous_right_fit is None):
        # Compute the histogram of the lower half image. It gives us the 2 pics where the lanes are located.
        histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
        # Separate the left part of the histogram from the right one. This is our starting point.
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base =  np.argmax(histogram[midpoint:]) + midpoint

        # Split the image in 9 horizontal strips
        n_windows = 9
        # Set height of windows
        window_height = int(img.shape[0]/n_windows)
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(n_windows):
            # Compute the windows boundaries
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_leftx_low = leftx_current - margin
            win_leftx_high = leftx_current + margin
            win_rightx_low = rightx_current - margin
            win_rightx_high = rightx_current + margin

            # Identify non zero pixels within left and right windows
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_leftx_low) & (nonzerox < win_leftx_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_rightx_low) & (nonzerox < win_rightx_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If non zeros pixels > minpix, recenter the next window on their mean
            if (len(good_left_inds) > minpix):
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if (len(good_right_inds) > minpix):
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Use last polynomial fit
    else:
        # Compute the windows boundaries
        previous_left_x = previous_left_fit[0]*nonzeroy**2 + previous_left_fit[1]*nonzeroy + previous_left_fit[2]
        win_leftx_low = previous_left_x - margin
        win_leftx_high =  previous_left_x + margin
        previous_right_x = previous_right_fit[0]*nonzeroy**2 + previous_right_fit[1]*nonzeroy + previous_right_fit[2]
        win_rightx_low = previous_right_x - margin
        win_rightx_high =  previous_right_x + margin
        # Identify non zero pixels within left and right windows
        good_left_inds = ((nonzerox >= win_leftx_low) & (nonzerox < win_leftx_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_rightx_low) & (nonzerox < win_rightx_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img = visualize_searching(img, nonzerox, nonzeroy, left_fit, right_fit, margin, left_lane_inds, right_lane_inds)

    return (out_img, left_fit, right_fit, left_lane_inds, right_lane_inds)

def visualize_searching(img, nonzerox, nonzeroy, left_fit, right_fit, margin, left_lane_inds, right_lane_inds):
    # Create an output image to draw on and visualize the result
    out_img = np.uint8(np.dstack((img, img, img))*255)
    window_img = np.zeros_like(out_img)

    # Color left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

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
    '''plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)'''

    return result

def compute_curvature_radius(img, left_fit, right_fit, left_lane_inds, right_lane_inds):
    # Get indices of all nonzero pixels along x and y axis
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_converted = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_converted = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Choose point to compute curvature just in front of the car
    yvalue = img.shape[0]

    # Compute curvature radius
    left_curv_radius = ((1 + (2*left_fit_converted[0]*yvalue + left_fit_converted[1])**2)**1.5) / (2*np.absolute(left_fit_converted[0]))
    right_curv_radius = ((1 + (2*right_fit_converted[0]*yvalue + right_fit_converted[1])**2)**1.5) / (2*np.absolute(right_fit_converted[0]))

    # Compute distance in meters of vehicle center from the line
    car_center = img.shape[1]/2  # we assume the camera is centered in the car
    lane_center = ((left_fit[0]*yvalue**2 + left_fit[1]*yvalue + left_fit[2]) + (right_fit[0]*yvalue**2 + right_fit[1]*yvalue + right_fit[2])) / 2
    center_dist = (lane_center - car_center) * xm_per_pix

    return (left_curv_radius, right_curv_radius, center_dist)

def draw_lane(img, warped, Minv, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result

def draw_data(img, top_img, bottom_img, left_curv_radius, right_curv_radius, center_dist):
    result = np.copy(img)

    # Take the mean of the left and right riadus
    mean_curv_radius = np.mean([left_curv_radius, right_curv_radius])

    # Add text to the original image
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(mean_curv_radius) + 'm'
    cv2.putText(result, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    if center_dist > 0:
        text = '{:04.2f}'.format(center_dist) + 'm left from the center'
    else:
        text = '{:04.2f}'.format(center_dist) + 'm right from the center'
    cv2.putText(result, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

    # Add transformed images to the original image
    mask = np.ones_like(top_img)*255
    img_1 = cv2.resize(top_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bottom_img_3_channels = np.uint8(np.dstack((bottom_img, bottom_img, bottom_img))*255)
    img_2 = cv2.resize(bottom_img_3_channels, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    offset = 50
    endy_img_1 = offset+img_1.shape[0]
    endy_img_2 = endy_img_1+img_2.shape[0]+20
    starty_img_2 = endy_img_1+20
    endx = top_img.shape[1]-offset
    startx = endx-img_1.shape[1]

    result[offset:endy_img_1, startx:endx] = img_1
    result[starty_img_2:endy_img_2, startx:endx] = img_2

    return result

def save_image_transform(original, transformed, is_gray, file_name):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=30)
    if (is_gray == True):
        ax2.imshow(transformed, cmap='gray')
    else:
        ax2.imshow(transformed)
    ax2.set_title('Result Image', fontsize=30)
    plt.savefig('./output_images/' + file_name + '.jpg')
    plt.show()

class Line:
    def __init__(self, max_lines=5):
        # Was the line detected in the last iteration?
        self.detected = False
        # Max number of last lines
        self.max_lines = max_lines
        # Polynomial coefficients for the most recent fit
        self.recent_fit = []
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None

    def reset(self):
        del self.recent_fit[:]
        self.best_fit = None
        self.detected = False

    def store_lines(self, left_fit, right_fit):
        if (left_fit is None or right_fit is None):
            self.detected = False
            # Remove the oldest fit from the history
            self.recent_fit.pop(0)
        else:
            self.detected = True
            if (len(self.recent_fit) == self.max_lines):
                # Remove the oldest fit from the history
                self.recent_fit.pop(0)
            # Add the new lines
            self.recent_fit.append((left_fit, right_fit))

        # Update best fit
        self.best_fit = np.average(self.recent_fit, axis=0)

    def process_img(self, img):
        ### 1. Distortion correction ###
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

        ### 3. Perspective transformation ###
        warped, Minv = perspective_transform(undistorted)

        ### 2. Gradient threshold ###
        gradient = color_gradient_threshold(warped)

        ### 4. Detect lines and store it ###
        if (self.detected):
            polyfit_image, left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_windows_polyfit(gradient, self.best_fit[0], self.best_fit[1])
        else:
            polyfit_image, left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_windows_polyfit(gradient)
        self.store_lines(left_fit, right_fit)

        ### 5. Visualize lane found back on to the road ###
        lanes = draw_lane(img, gradient, Minv, left_fit, right_fit)

        ### 6. Compute radius and display on image
        left_curv_radius, right_curv_radius, center_dist = compute_curvature_radius(gradient, left_fit, right_fit, left_lane_inds, right_lane_inds)
        return draw_data(lanes, polyfit_image, gradient, left_curv_radius, right_curv_radius, center_dist)

if __name__ == '__main__':
    ### Camera calibration ###
    if os.path.exists(calibration_file):
        print("Read in the calibration data")
        calibration_pickle = pickle.load(open(calibration_file, "rb"))
        mtx = calibration_pickle["mtx"]
        dist = calibration_pickle["dist"]
    else:
        print("Calibrate camera...")
        objpoints, imgpoints = get_points_for_calibration(9, 6)
        img = cv2.imread('./test_images/test1.jpg')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

        print("Save the camera calibration result for later use")
        calibration_pickle = {}
        calibration_pickle["mtx"] = mtx
        calibration_pickle["dist"] = dist
        pickle.dump(calibration_pickle, open(calibration_file, "wb"))

    # Create directory if it does not exist
    output_video_dir = "videos_output/"
    if not os.path.isdir(output_video_dir):
        os.makedirs(output_video_dir)

    video_output_1 = output_video_dir + 'project_video.mp4'
    video_output_2 = output_video_dir + 'challenge_video.mp4'

    print("Run pipeline for '" + video_output_1 + "'...")
    line = Line()
    video_input = VideoFileClip("project_video.mp4").subclip(0,2)
    processed_video = video_input.fl_image(line.process_img)
    processed_video.write_videofile(video_output_1, audio=False)

    '''print("Run pipeline for '" + video_output_2 + "'...")
    video_input = VideoFileClip("project_video.mp4")
    processed_video = video_input.fl_image(pipeline)
    processed_video.write_videofile(video_output_2, audio=False)'''
