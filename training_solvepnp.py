import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.model_selection import train_test_split


##TODO: Cross Validation

###DATA##
frameSize = (3976, 2652)
size_of_chessboard_squares_mm = 10
chessboardSize = (17, 28)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

#all_images = glob.glob("./data/test/*.jpg")
all_images = glob.glob("./data/links_8bit_jpg_DS_0.5/*.jpg")
#all_images = glob.glob("./data/rechts_8bit_jpg_DS_0.5/*.jpg")
# all_images = glob.glob("./data/rechts_8bit_jpg/*.jpg")

fold_train_errors = []
fold_test_errors = []

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
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    return ret, cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints
"""
###RPE###
def calculate_reprojection_error(objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist):
    mean_error = 0
    mean_errors_per_image = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        errors = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints[i])
        mean_error += errors
        mean_errors_per_image.append(errors)
    mean_error = mean_error / len(objpoints)
    return mean_errors_per_image, mean_error

###VALIDATION SOLVEPNP####
def validate_camera_calibration(test_images, objp, cameraMatrix, dist, criteria, chessboardSize):
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
def collect_rpe_incremental(images, test_size=0.2):
    train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
    total_train_images = len(train_images)
    num_images_list = []
    train_errors_list = []
    test_errors_list = []

    # Iterate over increasing sizes of training data
    for train_size in range(1, total_train_images + 1):
        current_train_imgs = train_images[:train_size]
        num_images_list.append(len(current_train_imgs))
        
        # Calibrate with current training set
        ret, cameraMatrix_train, dist_train, rvecs_train, tvecs_train, train_objpoints, train_imgpoints = calibrate_camera(current_train_imgs)
        
        # Calculate training reprojection error
        train_errors, mean_train_error = calculate_reprojection_error(train_objpoints, train_imgpoints, cameraMatrix_train, rvecs_train, tvecs_train, dist_train)
        
        # Validate with fixed test set
        test_errors, mean_test_error = validate_camera_calibration(test_images, objp, cameraMatrix_train, dist_train, criteria, chessboardSize)
        
        train_errors_list.append(mean_train_error)
        test_errors_list.append(mean_test_error)

    return num_images_list, train_errors_list, test_errors_list

num_images_list, train_errors_list, test_errors_list = collect_rpe_incremental(all_images, test_size=0.2)

plt.figure(figsize=(10, 6))
plt.plot(num_images_list, train_errors_list, 'o-', label='Train Reprojection Error')
plt.plot(num_images_list, test_errors_list, 'x--', label='Test Reprojection Error')
plt.xlabel('Number of Training Images')
plt.ylabel('Reprojection Error')
plt.title('Learning Curve for Camera Calibration')
plt.legend()
plt.grid(True)
plt.show()

##VISUALISATION###
# polynomial reg is not needed here because only 11 data points, still same trend as plotting with k fold 2
# plot with 60, 50 split and do poly
# Degree 7: Train MSE = 1.2460385480190124e-05, Validation MSE = 1.3188804584538585e-05
# polynomial regression or interpolation (not needed)

# polynomial regression
#compare_rpe(fold_train_errors, fold_test_errors, degree=9, interpolation=False)

# polynomial interpolation
# compare_rpe(fold_train_errors, fold_test_errors, interpolation=True)
"""
###LEARNING CURVE - KFOLD, TRAIN, TEST###

def collect_parameters_kfold_iterative(images, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    total_images = len(images)
    num_images_list = []
    train_params = []
    test_params = []

    for train_size in range(1, total_images + 1):
        if train_size < n_splits:
            continue

        current_train_imgs = images[:train_size]
        num_images_list.append(len(current_train_imgs))
        
        fold_train_results = []
        fold_test_results = []

        for train_index, test_index in kf.split(current_train_imgs):
            train_imgs = [current_train_imgs[i] for i in train_index]
            test_imgs = [current_train_imgs[i] for i in test_index]
            
            _, cameraMatrix_train, dist_train, rvecs_train, tvecs_train, _, _ = calibrate_camera(train_imgs)
            _, cameraMatrix_test, dist_test, rvecs_test, tvecs_test, _, _ = calibrate_camera(test_imgs)

            fold_train_results.append(
                {
                    "fx": cameraMatrix_train[0, 0],
                    "fy": cameraMatrix_train[1, 1],
                    "cx": cameraMatrix_train[0, 2],
                    "cy": cameraMatrix_train[1, 2],
                    "dist": dist_train.flatten(),
                    "rvecs": np.mean(np.array(rvecs_train), axis=0),
                    "tvecs": np.mean(np.array(tvecs_train), axis=0),
                }
            )

            fold_test_results.append(
                {
                    "fx": cameraMatrix_test[0, 0],
                    "fy": cameraMatrix_test[1, 1],
                    "cx": cameraMatrix_test[0, 2],
                    "cy": cameraMatrix_test[1, 2],
                    "dist": dist_test.flatten(),
                    "rvecs": np.mean(np.array(rvecs_test), axis=0),
                    "tvecs": np.mean(np.array(tvecs_test), axis=0),
                }
            )

        #avg folds
        avg_train_result = {
            key: np.mean([res[key] for res in fold_train_results], axis=0) for key in fold_train_results[0]
        }
        avg_test_result = {
            key: np.mean([res[key] for res in fold_test_results], axis=0) for key in fold_test_results[0]
        }

        train_params.append(avg_train_result)
        test_params.append(avg_test_result)

    return num_images_list, train_params, test_params

num_images_list, train_params, test_params = collect_parameters_kfold_iterative(all_images, n_splits=5)

fx_train = [params["fx"] for params in train_params]
fy_train = [params["fy"] for params in train_params]
cx_train = [params["cx"] for params in train_params]
cy_train = [params["cy"] for params in train_params]

fx_test = [params["fx"] for params in test_params]
fy_test = [params["fy"] for params in test_params]
cx_test = [params["cx"] for params in test_params]
cy_test = [params["cy"] for params in test_params]

rvecs_train = [params["rvecs"] for params in train_params]
tvecs_train = [params["tvecs"] for params in train_params]
rvecs_test = [params["rvecs"] for params in test_params]
tvecs_test = [params["tvecs"] for params in test_params]

dist_train = [params["dist"] for params in train_params]
dist_test = [params["dist"] for params in test_params]

fig, axs = plt.subplots(4, 1, figsize=(14, 24))

font_properties = {'fontsize': 7, 'fontweight': 'bold'}

#fx
axs[0].plot(num_images_list, fx_train, 'o-', label='fx Train')
axs[0].plot(num_images_list, fx_test, 'x--', label='fx Test')
axs[0].set_xlabel("Number of Training Images", **font_properties)
axs[0].set_ylabel("Parameter Value", **font_properties)
axs[0].set_title("Learning Curves for fx", **font_properties)
axs[0].legend(fontsize=7)

#fy
axs[1].plot(num_images_list, fy_train, 'o-', label='fy Train')
axs[1].plot(num_images_list, fy_test, 'x--', label='fy Test')
axs[1].set_xlabel("Number of Training Images", **font_properties)
axs[1].set_ylabel("Parameter Value", **font_properties)
axs[1].set_title("Learning Curves for fy", **font_properties)
axs[1].legend(fontsize=7)

#cx
axs[2].plot(num_images_list, cx_train, 'o-', label='cx Train')
axs[2].plot(num_images_list, cx_test, 'x--', label='cx Test')
axs[2].set_xlabel("Number of Training Images", **font_properties)
axs[2].set_ylabel("Parameter Value", **font_properties)
axs[2].set_title("Learning Curves for cx", **font_properties)
axs[2].legend(fontsize=7)

#cy
axs[3].plot(num_images_list, cy_train, 'o-', label='cy Train')
axs[3].plot(num_images_list, cy_test, 'x--', label='cy Test')
axs[3].set_xlabel("Number of Training Images", **font_properties)
axs[3].set_ylabel("Parameter Value", **font_properties)
axs[3].set_title("Learning Curves for cy", **font_properties)
axs[3].legend(fontsize=7)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(14, 12))

#rvecs
axs[0].plot(num_images_list, [rv[0] for rv in rvecs_train], 'o-', label='rvecs Train')
axs[0].plot(num_images_list, [rv[0] for rv in rvecs_test], 'x--', label='rvecs Test')
axs[0].set_xlabel("Number of Training Images", **font_properties)
axs[0].set_ylabel("Rotation Vector Value", **font_properties)
axs[0].set_title("Learning Curves for Rotation Vectors", **font_properties)
axs[0].legend(fontsize=7)

#tvecs
axs[1].plot(num_images_list, [tv[0] for tv in tvecs_train], 'o-', label='tvecs Train')
axs[1].plot(num_images_list, [tv[0] for tv in tvecs_test], 'x--', label='tvecs Test')
axs[1].set_xlabel("Number of Training Images", **font_properties)
axs[1].set_ylabel("Translation Vector Value", **font_properties)
axs[1].set_title("Learning Curves for Translation Vectors", **font_properties)
axs[1].legend(fontsize=7)

plt.tight_layout()
plt.show()

#dist coefficients
dist_labels = ["k1", "k2", "p1", "p2", "k3"]
font_properties = {'fontsize': 7, 'fontweight': 'bold'}

for i in range(5):
    plt.figure(figsize=(14, 6))
    plt.plot(num_images_list, [d[i] for d in dist_train], 'o-', label=f'{dist_labels[i]} Train')
    plt.plot(num_images_list, [d[i] for d in dist_test], 'x--', label=f'{dist_labels[i]} Test')
    plt.xlabel("Number of Training Images", **font_properties)
    plt.ylabel("Distortion Coefficient Value", **font_properties)
    plt.title(f"Learning Curves for {dist_labels[i]}", **font_properties)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()
