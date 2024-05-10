import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

""" Check overfitting by distances: 
Caliberation pattern that takes up less space in image (less information), might be more prone to overfiting ->  artificially low RPE vice versa.
 """

###DATA##
frameSize = (3976, 2652)
size_of_chessboard_squares_mm = 10
chessboardSize = (17, 28)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

all_images = glob.glob("links_8bit_jpg_DS_0.5/*.jpg")
# train_images = glob.glob('links_8bit_jpg_DS_0.5/*.jpg')
# test_images = glob.glob("test/*.jpg")

""" 
train_images, test_images = train_test_split(
    all_images, test_size=0.25, random_state=42
)
print(f"Training images: {len(train_images)}")
print(f"Testing images: {len(test_images)}") """


# Calculating distance by tvec: norm of tvec gives straight line distance from camera central point to caliberation pattern
def calibrate_camera(images, chessboardSize, square_size):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
        * square_size
    )

    objpoints = []
    imgpoints = []
    distances = []

    for image_path in images:
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    for i, corners2 in enumerate(imgpoints):
        _, rvec, tvec = cv.solvePnP(
            objpoints[i], corners2, cameraMatrix, dist
        )  # single image pose
        distances.append(np.linalg.norm(tvec))

    return cameraMatrix, dist, rvecs, tvecs, distances, objpoints, imgpoints


###RPE###
def calculate_reprojection_error(
    objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
):
    mean_error = 0
    mean_errors_per_image = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist
        )
        errors = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints[i])
        mean_error += errors
        mean_errors_per_image.append(errors)

    mean_error = mean_error / len(objpoints)
    return mean_errors_per_image, mean_error


def analyze_errors_by_distance(distances, errors):
    plt.scatter(
        distances,
        errors,
        c="blue",
        marker="o",
        label="Reprojection Error by Distance (mm)",
    )
    plt.xlabel("Distance from Camera")
    plt.ylabel("Reprojection Error")
    plt.title("Reprojection Error vs.Distance")
    plt.legend()
    plt.grid(True)
    plt.show()


cameraMatrix, dist, rvecs, tvecs, distances, objpoints, imgpoints = calibrate_camera(
    all_images, chessboardSize, size_of_chessboard_squares_mm
)

errors, mean_error = calculate_reprojection_error(
    objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
)

analyze_errors_by_distance(distances, errors)
