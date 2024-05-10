import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
##TODO: Cross Validation?

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

train_images, test_images = train_test_split(
    all_images, test_size=0.25, random_state=42
)
print(f"Training images: {len(train_images)}")
print(f"Testing images: {len(test_images)}")

##CALIBERATION###
def calibrate_camera(images):

    objpoints = []
    imgpoints = []

    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, frameSize, None, None
    )
    return ret, cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints


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


###VALIDATION SOLVEPNP####
""" Validate Intrinsic and Extrensic Parameters by using solvePNP on test data
Low RPE = Test, Train = caliberation parameters are reliable 
RPE < Train and > Test = Overfitting 
RPE > Train and < Test = Underfitting 
 """
##Not enough data available to produce sophisticated result

def validate_camera_calibration(
    test_images, objp, cameraMatrix, dist, criteria, chessboardSize
):
    errors = []
    rvecs, tvecs = [], []
    for image in test_images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret, rvec, tvec = cv.solvePnP(objp, corners2, cameraMatrix, dist)
            rvecs.append(rvec)  # new rvecs
            tvecs.append(tvec)  # new tvecs
            imgpoints2, _ = cv.projectPoints(objp, rvec, tvec, cameraMatrix, dist)
            error = cv.norm(corners2, imgpoints2, cv.NORM_L2) / len(corners2)
            errors.append(error)

    mean_error = np.mean(errors)
    return errors, mean_error


###CALL FUNC###
ret, cameraMatrix, dist, rvecs, tvecs, train_objpoints, train_imgpoints = (
    calibrate_camera(train_images)
)
train_errors, mean_train_error = calculate_reprojection_error(
    train_objpoints, train_imgpoints, cameraMatrix, rvecs, tvecs, dist
)

test_errors, mean_test_error = validate_camera_calibration(
    test_images, objp, cameraMatrix, dist, criteria, chessboardSize
)
print("Mean Reprojection Error for the Test Images:", test_errors)

##VISUALISATION
def compare_rpe(train_errors, test_errors, mean_train_error, mean_test_error):

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].bar(
        ["Train", "Test"], [mean_train_error, mean_test_error], color=["blue", "red"]
    )
    ax[0].set_title("Mean Reprojection Errors: Train vs Test")
    ax[0].set_ylabel("Mean Reprojection Error")

    max_length = max(len(train_errors), len(test_errors))
    x_train = np.arange(len(train_errors))
    x_test = np.arange(len(test_errors))

    ax[1].plot(
        x_train,
        train_errors,
        label="Train RPE",
        marker="o",
        linestyle="-",
        color="blue",
    )
    ax[1].plot(
        x_test, test_errors, label="Test RPE", marker="o", linestyle="-", color="red"
    )
    ax[1].set_title("Reprojection Errors Over Images")
    ax[1].set_xlabel("Image Index")
    ax[1].set_ylabel("Reprojection Error")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


compare_rpe(train_errors, test_errors, mean_train_error, mean_test_error)
