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
    direction = np.arctan2(abs_sobely, abs_sobely)
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

def region_of_interest(img):
    # Set vertices for the mask
    imshape = img.shape # x: imshape[1], y: imshape[0]
    left_bottom = (0.15*imshape[1], imshape[0])
    left_top = (0.45*imshape[1], 0.6*imshape[0])
    right_top = (0.60*imshape[1], 0.6*imshape[0])
    right_bottom = (0.9*imshape[1], imshape[0])
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    # Define blank mask to start with
    mask = np.zeros_like(img)

    # Define a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on the image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def perspective_transform(img):
    # From trapezoidale shape on straight lines...
    src = np.float32([[519, 502], [765, 502], [1029, 668], [275, 668]])
    # ...to rectangle
    dst = np.float32([[275, 502], [1029, 502], [1029, 668], [275, 668]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    return (cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR), Minv)

def sliding_windows_polyfit(img):

    # Compute the histogram of the lower half image. It gives us the 2 pics where the lanes are located.
    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.uint8(np.dstack((img, img, img))*255)
    # Separate the left part of the histogram from the right one. This is our starting point.
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base =  np.argmax(histogram[midpoint:]) + midpoint

    # Split the image in 9 horizontal strips
    n_windows = 9
    # Set height of windows
    window_height = int(img.shape[0]/n_windows)
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Get indices of all nonzero pixels along x and y axis
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Compute the windows boundaries
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_leftx_low = leftx_current - margin
        win_leftx_high = leftx_current + margin
        win_rightx_low = rightx_current - margin
        win_rightx_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_leftx_low,win_y_low), (win_leftx_high,win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_rightx_low,win_y_low), (win_rightx_high,win_y_high), (0,255,0), 2)

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

    # Display left line in red, and right line in blue
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    '''plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('./output_images/sliding_windows_polyfit.jpg')
    plt.show()'''

    return (out_img, ploty, left_fitx, right_fitx, leftx, rightx, lefty, righty)

def compute_curvature_radius(img, leftx, rightx, lefty, righty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Choose point to compute curvature just in front of the car
    yvalue = img.shape[0]

    # Compute curvature radius
    left_curv_radius = ((1 + (2*left_fit[0]*yvalue + left_fit[1])**2)**1.5) / (2*np.absolute(left_fit[0]))
    right_curv_radius = ((1 + (2*right_fit[0]*yvalue + right_fit[1])**2)**1.5) / (2*np.absolute(right_fit[0]))
    return (left_curv_radius, right_curv_radius)

def draw_lane(img, warped, Minv, ploty, left_fitx, right_fitx):
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

def draw_data(img, polyfit_img, gradient_binary, left_curv_radius, right_curv_radius):
    result = np.copy(img)

    # Take the mean of the left and right riadus
    mean_curv_radius = np.mean([left_curv_radius, right_curv_radius])

    # Add text to the original image
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(mean_curv_radius) + 'm'
    cv2.putText(result, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

    # Add transformed images to the original image
    mask = np.ones_like(polyfit_img)*255
    img_1 = cv2.resize(polyfit_img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    gradient = np.uint8(np.dstack((gradient_binary, gradient_binary, gradient_binary))*255)
    img_2 = cv2.resize(gradient, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

    offset = 50
    endy_img_1 = offset+img_1.shape[0]
    endy_img_2 = endy_img_1+img_2.shape[0]+20
    starty_img_2 = endy_img_1+20
    endx = polyfit_img.shape[1]-offset
    startx = endx-img_1.shape[1]

    mask[offset:endy_img_1, startx:endx] = img_1
    mask[starty_img_2:endy_img_2, startx:endx] = img_2

    return cv2.bitwise_and(result, mask)

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

def pipeline(img):
    ### 1. Distortion correction ###
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    ### 2. Gradient threshold ###
    gradient = color_gradient_threshold(undistorted)
    ### 2. Region of interest ###
    masked_image = region_of_interest(gradient)
    ### 3. Perspective transformation ###
    warped, Minv = perspective_transform(masked_image)
    ### 4. Detect lines ###
    polyfit_image, ploty, left_fitx, right_fitx, leftx, rightx, lefty, righty = sliding_windows_polyfit(warped)
    ### 5. Visualize lane founded back on to the road ###
    lanes = draw_lane(img, warped, Minv, ploty, left_fitx, right_fitx)
    ### 6. Compute radius and display on image
    left_curv_radius, right_curv_radius = compute_curvature_radius(lanes, leftx, rightx, lefty, righty)
    return draw_data(lanes, polyfit_image, gradient, left_curv_radius, right_curv_radius)

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

    print("Run pipeline for '" + video_output_1 + "' video...")
    video_input = VideoFileClip("project_video.mp4")
    processed_video = video_input.fl_image(pipeline)
    processed_video.write_videofile(video_output_1, audio=False)

    print("Run pipeline for '" + video_output_2 + "' video...")
    video_input = VideoFileClip("project_video.mp4")
    processed_video = video_input.fl_image(pipeline)
    processed_video.write_videofile(video_output_2, audio=False)
