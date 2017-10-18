import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import os

# Parameters for calibration
mtx = []
dist = []
calibration_file = "calibration_pickle.p"

# Parameters for gradient threshold
ksize = 3
gradx_thresh = (170, 255)
grady_thresh = (170, 255)
magni_thresh = (30, 255)
dir_thresh = (0.25, 1.0)

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

def gradient_threshold(img):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=gradx_thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=grady_thresh)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=magni_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)

    # Combine threshold
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined

def perspective_transform(img):
    # From trapezoidale shape on straight lines...
    src = np.float32([[610, 439], [670, 439], [1029, 668], [275, 668]])
    # ...to rectangle
    dst = np.float32([[275, 439], [1029, 439], [1029, 668], [275, 668]])
    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

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

def pipeline(img):
    ### 1. Distortion correction ###
    print('Undistort image...')
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    save_image_transform(img, undistorted, False, 'undistorted')

    ### 2. Gradient threshold ###
    print('Apply gradient threshold...')
    gradient = gradient_threshold(undistorted)
    save_image_transform(img, gradient, True, 'gradient')

    ### 3. Perspective transformation ###
    print('Apply perspective transform...')
    perspective = perspective_transform(undistorted)
    save_image_transform(img, perspective, False, 'perspective')

    print('Done!')
    return perspective

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

    ### Apply pipeline ###
    img = cv2.imread('./test_images/straight_lines1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pipeline(img)
