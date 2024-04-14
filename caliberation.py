###Viktoria Dergunova

import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from datetime import datetime

chessboardSize = (17, 28) # -1
frameSize = (3976, 2652)

#####CALIBERATION######

# Termination criteria
#num of iteration: increase -> more refined results
#num of epsilon: drecrease -> tightens the convergence criteria,refines the corner locations to a finer accuracy


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
size_of_chessboard_squares_mm = 10
objp *= size_of_chessboard_squares_mm

objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob('links_8bit_jpg_DS_0.5/*.jpg') ##PATH

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # gray = cv.equalizeHist(gray)  # Equalize histogram to enhance contrast
    # gray = cv.GaussianBlur(gray, (5, 5), 0) #Reduce noise

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        objpoints.append(objp)
        #subpixel
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

#print("Distortion Coefficients Shape:", dist.shape)
#print("Distortion Coefficients:", dist)

# WRITE XML
current_date = datetime.now().strftime("%Y-%m-%d")

fs = cv.FileStorage("calibration.xml", cv.FILE_STORAGE_WRITE)
fs.write("calibration_date", current_date)
fs.startWriteStruct("camera_matrix", cv.FileNode_MAP)
# Intrinsic Parameters
fs.write("fx", cameraMatrix[0, 0]) # focal length of camera, x (pixel unit)
fs.write("fy", cameraMatrix[1, 1]) # focal length of camera, y (pixel unit)
fs.write("cx", cameraMatrix[0, 2]) # optical center coordinates of camera, x (pixel unit)
fs.write("cy", cameraMatrix[1, 2]) # optical center coordinates of camera, y (pixel unit)
fs.endWriteStruct()

fs.startWriteStruct("distortion_coefficients", cv.FileNode_MAP)
fs.write("k1", dist[0, 0]) #radial distortion coefficients
fs.write("k2", dist[0, 1]) #radial distortion coefficients
fs.write("k3", dist[0, 4]) #radial distortion coefficients, correction for severe distortion
fs.write("p1", dist[0, 2]) #tangential distortion coefficients
fs.write("p2", dist[0, 3]) #tangential distortion coefficients
fs.endWriteStruct()

fs.write("image_height", 3976)
fs.write("image_width", 2652)

fs.release()

print("Camera matrix:\n", cameraMatrix)
print("Distortion coefficients:\n", dist.flatten())

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
plt.title('RPE')
plt.xlabel('Image Index')
plt.ylabel('Reprojection Error')
plt.legend()
plt.grid(True)
plt.savefig('residual_plot_rpe.png')
plt.show()


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

    # absolute Value of RPEs
    ax1 = plt.subplot(2, 2, 1)
    hb1 = plt.hexbin(error_x, error_y, C=error_magnitude, gridsize=30, cmap='coolwarm', bins='log')
    plt.colorbar(hb1, ax=ax1)
    plt.title('Absolute Value of RPEs')
    plt.xlabel('x-component')
    plt.ylabel('y-component')
    ax1.set_xlim([min(error_x), max(error_x)])
    ax1.set_ylim([min(error_y), max(error_y)])

    # direction of RPE Vectors
    ax2 = plt.subplot(2, 2, 2)
    hb2 = plt.hexbin(error_x, error_y, C=error_direction, gridsize=30, cmap='coolwarm')
    plt.colorbar(hb2, ax=ax2)
    plt.title('Direction of RPE vectors')
    plt.xlabel('x-component')
    plt.ylabel('y-component')
    ax2.set_xlim([min(error_x), max(error_x)])
    ax2.set_ylim([min(error_y), max(error_y)])

    # distribution of RPEs
    ax3 = plt.subplot(2, 2, 3)
    hb3 = plt.hist2d(error_x, error_y, bins=50, cmap='plasma')
    plt.colorbar(hb3[3], ax=ax3)
    plt.title('Distribution of RPEs')
    plt.xlabel('x-component')
    plt.ylabel('y-component')
    ax3.set_xlim([min(error_x), max(error_x)])
    ax3.set_ylim([min(error_y), max(error_y)])

    # histogram of Absolute RPEs
    ax4 = plt.subplot(2, 2, 4)
    plt.hist(error_magnitude, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Absolute RPEs')
    plt.xlabel('Absolute RPE')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('rpe_analysis.png')
    plt.show()

visualize_rpes(error_x, error_y)

