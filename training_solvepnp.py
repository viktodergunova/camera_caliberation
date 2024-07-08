import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns


##TODO: Cross Validation

###DATA##
frameSize = (3976, 2652)
size_of_chessboard_squares_mm = 10
chessboardSize = (17, 28)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

all_images = glob.glob("./data/test/*.jpg")
#all_images = glob.glob("./data/links_8bit_jpg_DS_0.5/*.jpg")

# all_images = glob.glob("./data/links_8bit_jpg/*.jpg")
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
## Cross-Validation Loop ##
""" kf = KFold(n_splits=2, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(all_images)):
    train_images = [all_images[i] for i in train_idx]
    test_images = [all_images[i] for i in test_idx]

    ret, cameraMatrix, dist, rvecs, tvecs, train_objpoints, train_imgpoints = calibrate_camera(train_images)

    train_errors, mean_train_error = calculate_reprojection_error(
        train_objpoints, train_imgpoints, cameraMatrix, rvecs, tvecs, dist
    )

    test_errors, mean_test_error = validate_camera_calibration(
        test_images, objp, cameraMatrix, dist, criteria, chessboardSize
    )

    # fold_mean_train_errors.append(mean_train_error)
    # fold_mean_test_errors.append(mean_test_error)

    fold_train_errors.append(train_errors)
    fold_test_errors.append(test_errors)

    print(f"Fold {fold + 1}: Mean Train Error = {mean_train_error}, Mean Test Error = {mean_test_error}")


##VISUALISATION###
# polynomial reg is not needed here because only 11 data points, still same trend as plotting with k fold 2
# plot with 60, 50 split and do poly
# Degree 7: Train MSE = 1.2460385480190124e-05, Validation MSE = 1.3188804584538585e-05
# polynomial regression or interpolation (not needed)
def compare_rpe(train_errors, test_errors, degree=None, interpolation=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    flat_train = [item for sublist in train_errors for item in sublist]
    flat_test = [item for sublist in test_errors for item in sublist]

    train_indices = np.arange(len(flat_train))
    test_indices = np.arange(len(flat_test))

    if interpolation:
        train_poly = np.polyfit(train_indices, flat_train, len(train_indices) - 1)
        test_poly = np.polyfit(test_indices, flat_test, len(test_indices) - 1)
    else:
        train_poly = np.polyfit(train_indices, flat_train, degree)
        test_poly = np.polyfit(test_indices, flat_test, degree)

    train_poly_y = np.polyval(train_poly, train_indices)
    test_poly_y = np.polyval(test_poly, test_indices)

    ax.plot(train_indices, flat_train, marker="o", color="blue", linestyle="None", label="Training Dataset RPE")
    ax.plot(test_indices, flat_test, marker="x", color="red", linestyle="None", label="Test Dataset RPE")
    ax.plot(
        train_indices,
        train_poly_y,
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Training Dataset RPE: Polynomial Degree 3",
    )
    ax.plot(
        test_indices,
        test_poly_y,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Test Dataset RPE: Polynomial Degree 3",
    )

    ax.set_xlabel("Fold Index")
    ax.set_ylabel("Reprojection Error")
    ax.set_ylim([0, 0.030])
    ax.set_title("Reprojection Errors for Training and Test Datasets")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

 """
# polynomial regression
#compare_rpe(fold_train_errors, fold_test_errors, degree=9, interpolation=False)

# polynomial interpolation
# compare_rpe(fold_train_errors, fold_test_errors, interpolation=True)

###LEARNING CURVE - KFOLD, TRAIN, TEST###

def collect_parameters_kfold_iterative(images, n_splits=2):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    total_images = len(images)
    num_images_list = []
    train_results = []
    test_results = []

    # Iterative collection
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

        train_results.append(avg_train_result)
        test_results.append(avg_test_result)

    return num_images_list, train_results, test_results

num_images_list, train_results, test_results = collect_parameters_kfold_iterative(all_images, n_splits=2)

fx_train = [result["fx"] for result in train_results]
fy_train = [result["fy"] for result in train_results]
cx_train = [result["cx"] for result in train_results]
cy_train = [result["cy"] for result in train_results]

fx_test = [result["fx"] for result in test_results]
fy_test = [result["fy"] for result in test_results]
cx_test = [result["cx"] for result in test_results]
cy_test = [result["cy"] for result in test_results]

rvecs_train = [result["rvecs"] for result in train_results]
tvecs_train = [result["tvecs"] for result in train_results]
rvecs_test = [result["rvecs"] for result in test_results]
tvecs_test = [result["tvecs"] for result in test_results]

dist_train = [result["dist"] for result in train_results]
dist_test = [result["dist"] for result in test_results]

# Plotting
fig, axs = plt.subplots(11, 1, figsize=(14, 66))  # Adjusted to 11 subplots

axs[0].plot(num_images_list, fx_train, 'o-', label='fx Train')
axs[0].plot(num_images_list, fx_test, 'x--', label='fx Test')
axs[0].set_xlabel("Number of Training Images")
axs[0].set_ylabel("Parameter Value")
axs[0].set_title("Learning Curves for fx")
axs[0].legend()

axs[1].plot(num_images_list, fy_train, 'o-', label='fy Train')
axs[1].plot(num_images_list, fy_test, 'x--', label='fy Test')
axs[1].set_xlabel("Number of Training Images")
axs[1].set_ylabel("Parameter Value")
axs[1].set_title("Learning Curves for fy")
axs[1].legend()

axs[2].plot(num_images_list, cx_train, 'o-', label='cx Train')
axs[2].plot(num_images_list, cx_test, 'x--', label='cx Test')
axs[2].set_xlabel("Number of Training Images")
axs[2].set_ylabel("Parameter Value")
axs[2].set_title("Learning Curves for cx")
axs[2].legend()

axs[3].plot(num_images_list, cy_train, 'o-', label='cy Train')
axs[3].plot(num_images_list, cy_test, 'x--', label='cy Test')
axs[3].set_xlabel("Number of Training Images")
axs[3].set_ylabel("Parameter Value")
axs[3].set_title("Learning Curves for cy")
axs[3].legend()

axs[4].plot(num_images_list, [rv[0] for rv in rvecs_train], 'o-', label='rvecs Train')
axs[4].plot(num_images_list, [rv[0] for rv in rvecs_test], 'x--', label='rvecs Test')
axs[4].set_xlabel("Number of Training Images")
axs[4].set_ylabel("Rotation Vector Value")
axs[4].set_title("Learning Curves for Rotation Vectors")
axs[4].legend()

axs[5].plot(num_images_list, [tv[0] for tv in tvecs_train], 'o-', label='tvecs Train')
axs[5].plot(num_images_list, [tv[0] for tv in tvecs_test], 'x--', label='tvecs Test')
axs[5].set_xlabel("Number of Training Images")
axs[5].set_ylabel("Translation Vector Value")
axs[5].set_title("Learning Curves for Translation Vectors")
axs[5].legend()

dist_labels = ["k1", "k2", "p1", "p2", "k3"]
for i in range(len(dist_labels)):
    axs[6 + i].plot(num_images_list, [d[i] for d in dist_train], 'o-', label=f'{dist_labels[i]} Train')
    axs[6 + i].plot(num_images_list, [d[i] for d in dist_test], 'x--', label=f'{dist_labels[i]} Test')
    axs[6 + i].set_xlabel("Number of Training Images")
    axs[6 + i].set_ylabel("Distortion Coefficient Value")
    axs[6 + i].set_title(f"Learning Curves for {dist_labels[i]}")
    axs[6 + i].legend()

plt.tight_layout()
plt.show()