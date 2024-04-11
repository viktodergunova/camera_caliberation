###Viktoria Dergunova

import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

chessboardSize = (17, 28) # -1
frameSize = (3976, 2652)

#####CALIBERATION######

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
size_of_chessboard_squares_mm = 10
objp *= size_of_chessboard_squares_mm

objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob('rechts_8bit_jpg_DS_0.5/*.jpg') ##PATH

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# save camera calibration result as XML
fs = cv.FileStorage("calibration.xml", cv.FILE_STORAGE_WRITE)
fs.write("camera_matrix", cameraMatrix)
fs.write("distortion_coefficients", dist)
fs.write("image_height", frameSize[1])
fs.write("image_width", frameSize[0])
fs.release()


print("Camera matrix:\n", cameraMatrix)
print("Distortion coefficients:\n", dist.flatten())

# distortion coefficients 
print("Distortion coefficients:", dist.flatten())

# compute reprojection error
def compute_reprojection_error(objpoints, imgpoints, cameraMatrix, dist,rvecs,tvecs):
    mean_error = 0
    total_points = 0
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
        total_points += 1
        errors.append(error)
    mean_error /= total_points
    return mean_error, errors


reprojection_error, errors = compute_reprojection_error(objpoints, imgpoints, cameraMatrix, dist,rvecs, tvecs)
print("Reprojection error:", reprojection_error)


###############VISUALIZATION###########

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.plot(errors, marker='o', linestyle='',label='Individuel Reprojection Error')
plt.axhline(y=reprojection_error, color='r', linestyle='-', label='Mean Reprojection Error')
plt.title('Residual Plot: RPE')
plt.xlabel('Image Index')
plt.ylabel('Reprojection Error')
plt.legend()
plt.grid(True)
plt.savefig('residual_plot_rpe.png')
plt.show()

h, w = frameSize

y, x = np.mgrid[0:h:1, 0:w:1].astype(np.float32)
pixel_grid = np.column_stack((x.ravel(), y.ravel()))

pixel_grid = pixel_grid.reshape(-1, 1, 2)
normalized_grid = cv.undistortPoints(pixel_grid, cameraMatrix, dist, None, cameraMatrix)

map_x, map_y = normalized_grid[:,0,0], normalized_grid[:,0,1]
map_x = map_x.reshape(h, w)
map_y = map_y.reshape(h, w)

fig, ax = plt.subplots(figsize=(10, 6))


spacing = 100 
for i in range(0, h, spacing):
    ax.plot(range(w), [i] * w, color='blue', linestyle='--')  # horizontal lines, distorted
for j in range(0, w, spacing):
    ax.plot([j] * h, range(h), color='blue', linestyle='--')  # vertical lines

#undistorted grid lines
for i in range(0, h, spacing):
    ax.plot(map_x[i, :], map_y[i, :], color='red')  # horizontal lines, undistorted
for j in range(0, w, spacing):
    ax.plot(map_x[:, j], map_y[:, j], color='red')  # vertical lines

ax.set_xlim([0, w])
ax.set_ylim([h, 0])
ax.set_title('Grid Distortion and Correction Comparison')
ax.set_xlabel('u (along X axis with {} pixels)'.format(w))
ax.set_ylabel('v (along Y axis with {} pixels)'.format(h))


plt.savefig('distortion_visualization.png')
plt.show()

# Undistorted Image Comparison
def compare_images(orig_img, undist_img):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(cv.cvtColor(orig_img, cv.COLOR_BGR2RGB))
    axs[0].set_title('Original Distorted Image')
    axs[0].axis('off')
    axs[1].imshow(cv.cvtColor(undist_img, cv.COLOR_BGR2RGB))
    axs[1].set_title('Undistorted Image')
    axs[1].axis('off')
    plt.savefig('distortion_vs.undistortion_img.png')
    plt.show()

# Load distorted image
distorted_img = cv.imread('rechts_8bit_jpg_DS_0.5/DSC00003_DS_0.5.jpg') ###PATH
if distorted_img is None:
    print("Error: Failed to load the image.")
else:
    # Undistort the image
    undistorted_img = cv.undistort(distorted_img, cameraMatrix, dist)
    compare_images(distorted_img, undistorted_img)

# compute_reprojection_error to return detailed error components for x and y for coefficient calculation
def compute_detailed_reprojection_error(objpoints, imgpoints, cameraMatrix, dist):
    mean_error = 0
    total_points = 0
    errors = []
    error_x = []
    error_y = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        errors_vector = imgpoints[i] - imgpoints2
        error_x.extend(errors_vector[:,:,0].flatten())
        error_y.extend(errors_vector[:,:,1].flatten())
        mean_error += error
        total_points += len(objpoints[i])
        errors.append(error)
    mean_error /= total_points
    return mean_error, errors, error_x, error_y

# reprojection error with x and y  
detailed_reprojection_error, errors, error_x, error_y = compute_detailed_reprojection_error(objpoints, imgpoints, cameraMatrix, dist)

def visualize_rpes(error_x, error_y):
    # calculate the absolute RPE and direction for each point
    error_magnitude = np.sqrt(np.array(error_x)**2 + np.array(error_y)**2)
    error_direction = np.arctan2(error_y, error_x)

    plt.figure(figsize=(20, 10))

    # Absolute Value of RPEs
    ax1 = plt.subplot(2, 2, 1)
    hb1 = plt.hexbin(error_x, error_y, C=error_magnitude, gridsize=30, cmap='viridis', bins='log')
    plt.colorbar(hb1, ax=ax1)
    plt.title('Absolute Value of RPEs')
    plt.xlabel('x-component')
    plt.ylabel('y-component')
    ax1.set_xlim([min(error_x), max(error_x)])
    ax1.set_ylim([min(error_y), max(error_y)])

    # Direction of RPE Vectors
    ax2 = plt.subplot(2, 2, 2)
    hb2 = plt.hexbin(error_x, error_y, C=error_direction, gridsize=30, cmap='coolwarm')
    plt.colorbar(hb2, ax=ax2)
    plt.title('Direction of RPE vectors')
    plt.xlabel('x-component')
    plt.ylabel('y-component')
    ax2.set_xlim([min(error_x), max(error_x)])
    ax2.set_ylim([min(error_y), max(error_y)])

    # Distribution of RPEs
    ax3 = plt.subplot(2, 2, 3)
    hb3 = plt.hist2d(error_x, error_y, bins=50, cmap='plasma')
    plt.colorbar(hb3[3], ax=ax3)
    plt.title('Distribution of RPEs')
    plt.xlabel('x-component')
    plt.ylabel('y-component')
    ax3.set_xlim([min(error_x), max(error_x)])
    ax3.set_ylim([min(error_y), max(error_y)])

    # Histogram of Absolute RPEs
    ax4 = plt.subplot(2, 2, 4)
    plt.hist(error_magnitude, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Absolute RPEs')
    plt.xlabel('Absolute RPE')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('rpe_analysis.png')
    plt.show()

visualize_rpes(error_x, error_y)

#####VISUALIZATION COEFFICIENTS#####

fs_read = cv.FileStorage("calibration.xml", cv.FILE_STORAGE_READ)
cameraMatrix = fs_read.getNode("camera_matrix").mat()
distCoeffs = fs_read.getNode("distortion_coefficients").mat()
fs_read.release()


print("Camera matrix:\n", cameraMatrix)
print("Distortion coefficients:\n", distCoeffs.flatten())

# grid of points
step = 20
w, h = 3976, 2652 
x, y = np.meshgrid(range(0, w, step), range(0, h, step))
pts = np.vstack((x.flatten(), y.flatten())).astype(np.float32).T


def draw_grid(img, pts, color=(255,0, 0)):  #green
    for i in range(pts.shape[0]):
        pt = tuple(pts[i].astype(int))
        cv.circle(img, pt, 3, color, -1)

original_distCoeffs = distCoeffs.flatten()

coeff_sets = [original_distCoeffs,  # Original coefficients
              np.array([original_distCoeffs[0], original_distCoeffs[1], 0, 0, 0]), #radial distortion
              ]
for idx, coeffs in enumerate(coeff_sets):
   
    img = np.zeros((h, w, 3), dtype=np.uint8)


    draw_grid(img, pts, color=(255, 255, 255))  # Original grid in blue

    # apply distortion/undistortion to get the new grid points
    pts_distorted = cv.undistortPoints(np.expand_dims(pts, axis=1), cameraMatrix, coeffs, None, cameraMatrix)
    pts_distorted = pts_distorted.reshape(-1, 2)

    # distorted/undistorted grid in blue
    draw_grid(img, pts_distorted, color=(0, 0, 255))  # Distorted/Undistorted grid in red

    coeffs_str = '_'.join([f"{c:.2e}" for c in coeffs]).replace('.', 'p').replace('-', 'm').replace('+', '')
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Distortion with coefficients: {coeffs}")
    plt.axis('off')
    plt.savefig(f'distortion_visualization_{idx}_{coeffs_str}.png')
    plt.close()
