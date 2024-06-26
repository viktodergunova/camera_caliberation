import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns


##TODO: Cross Validat

###DATA##
frameSize = (3976, 2652)
size_of_chessboard_squares_mm = 10
chessboardSize = (17, 28)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

# all_images = glob.glob("./data/test/*.jpg")
all_images = glob.glob("./data/rechts_8bit_jpg_DS_0.5/*.jpg")

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
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(all_images)):
    train_images = [all_images[i] for i in train_idx]
    test_images = [all_images[i] for i in test_idx]

    # Calibration using training images
    ret, cameraMatrix, dist, rvecs, tvecs, train_objpoints, train_imgpoints = calibrate_camera(train_images)

    # Calculate reprojection error for training
    train_errors, mean_train_error = calculate_reprojection_error(
        train_objpoints, train_imgpoints, cameraMatrix, rvecs, tvecs, dist
    )

    # Validation using test images
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

    ax.set_xlabel("Image Index")
    ax.set_ylabel("Reprojection Error")
    ax.set_ylim([0, 0.030])
    ax.set_title("Reprojection Errors for Training and Test Datasets")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# polynomial regression
compare_rpe(fold_train_errors, fold_test_errors, degree=9, interpolation=False)

# polynomial interpolation
# compare_rpe(fold_train_errors, fold_test_errors, interpolation=True)


###LEARNING CURVE - ITERATIVE APPROACH ###
"""
def collect_parameters(images, objp, chessboardSize, criteria):
    fx_values, fy_values, cx_values, cy_values = [], [], [], []
    rvecs_values, tvecs_values = [], []
    fx_std, fy_std, cx_std, cy_std = [], [], [], []
    rvecs_std, tvecs_std = [], []
    dist_values, dist_std = [], []
    num_images = range(1, len(images) + 1)
    for n in num_images:
        subset_images = images[:n]
        _, cameraMatrix, dist, rvecs, tvecs, _, _ = calibrate_camera(subset_images)
        fx_values.append(cameraMatrix[0, 0])
        fy_values.append(cameraMatrix[1, 1])
        cx_values.append(cameraMatrix[0, 2])
        cy_values.append(cameraMatrix[1, 2])
        rvecs_values.append(np.mean(np.array(rvecs), axis=0))
        tvecs_values.append(np.mean(np.array(tvecs), axis=0))
        dist_values.append(dist.flatten())

        if n > 1:
            fx_std.append(np.std(fx_values))
            fy_std.append(np.std(fy_values))
            cx_std.append(np.std(cx_values))
            cy_std.append(np.std(cy_values))
            rvecs_std.append(np.std([rv[0] for rv in rvecs_values]))
            tvecs_std.append(np.std([tv[0] for tv in tvecs_values]))
            dist_std.append(np.std(np.array(dist_values), axis=0))
        else:
            fx_std.append(0)
            fy_std.append(0)
            cx_std.append(0)
            cy_std.append(0)
            rvecs_std.append(0)
            tvecs_std.append(0)
            dist_std.append(np.zeros_like(dist.flatten()))

    return num_images, fx_values, fy_values, cx_values, cy_values, rvecs_values, tvecs_values, dist_values, fx_std, fy_std, cx_std, cy_std, rvecs_std, tvecs_std, dist_std

num_images, fx_values, fy_values, cx_values, cy_values, rvecs_values, tvecs_values, dist_values, fx_std, fy_std, cx_std, cy_std, rvecs_std, tvecs_std, dist_std = collect_parameters(all_images, objp, chessboardSize, criteria)

fig, axs = plt.subplots(2, 1, figsize=(14, 12))

degree = 3
fx_poly = np.polyfit(num_images, fx_values, degree)
fy_poly = np.polyfit(num_images, fy_values, degree)
cx_poly = np.polyfit(num_images, cx_values, degree)
cy_poly = np.polyfit(num_images, cy_values, degree)

fx_poly_y = np.polyval(fx_poly, num_images)
fy_poly_y = np.polyval(fy_poly, num_images)
cx_poly_y = np.polyval(cx_poly, num_images)
cy_poly_y = np.polyval(cy_poly, num_images)

axs[0].errorbar(num_images, cx_values, yerr=cx_std, fmt='o-', label='cx', capsize=5)
axs[0].errorbar(num_images, cy_values, yerr=cy_std, fmt='o-', label='cy', capsize=5)
axs[0].plot(num_images, cx_poly_y, '-', label='cx Poly Fit')
axs[0].plot(num_images, cy_poly_y, '-', label='cy Poly Fit')

axs[0].set_xlabel('Number of Calibration Images')
axs[0].set_ylabel('Parameter Value')
axs[0].set_title('Learning Curves for cx and cy/ Std Deviation')
axs[0].legend()

axs[1].errorbar(num_images, fx_values, yerr=fx_std, fmt='o-', label='fx', capsize=5)
axs[1].errorbar(num_images, fy_values, yerr=fy_std, fmt='o-', label='fy', capsize=5)
axs[1].plot(num_images, fx_poly_y, '-', label='fx Poly Fit')
axs[1].plot(num_images, fy_poly_y, '-', label='fy Poly Fit')

axs[1].set_xlabel('Number of Calibration Images')
axs[1].set_ylabel('Parameter Value')
axs[1].set_title('Learning Curves for fx and fy /Std Deviation')
axs[1].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(14, 12))

rvecs_poly = np.polyfit(num_images, [rv[0][0] for rv in rvecs_values], degree)
tvecs_poly = np.polyfit(num_images, [tv[0][0] for tv in tvecs_values], degree)

rvecs_poly_y = np.polyval(rvecs_poly, num_images)
tvecs_poly_y = np.polyval(tvecs_poly, num_images)


axs[0].errorbar(num_images, [rv[0][0] for rv in rvecs_values], yerr=rvecs_std, fmt='o-', label='rvec', capsize=5)
axs[0].plot(num_images, rvecs_poly_y, '-', label='rvec Poly Fit')
axs[0].set_xlabel('Number of Calibration Images')
axs[0].set_ylabel('Rotation Vector Value')
axs[0].set_title('Learning Curves for Rotation Vectors/ Std Deviation')
axs[0].legend()

axs[1].errorbar(num_images, [tv[0][0] for tv in tvecs_values], yerr=tvecs_std, fmt='o-', label='tvec', capsize=5)
axs[1].plot(num_images, tvecs_poly_y, '-', label='tvec Poly Fit')
axs[1].set_xlabel('Number of Calibration Images')
axs[1].set_ylabel('Translation Vector Value')
axs[1].set_title('Learning Curves for Translation/Std Deviation')
axs[1].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(5, 1, figsize=(14, 30))

dist_labels = ['k1', 'k2', 'p1', 'p2', 'k3']
for i in range(5):
    dist_poly = np.polyfit(num_images, [d[i] for d in dist_values], degree)
    dist_poly_y = np.polyval(dist_poly, num_images)
    
    dist_std_i = np.array([std[i] for std in dist_std])
    
    axs[i].errorbar(num_images, [d[i] for d in dist_values], yerr=dist_std_i, fmt='o-', label=dist_labels[i], capsize=5)
    axs[i].plot(num_images, dist_poly_y, '-', label=f'{dist_labels[i]} Poly Fit')
    axs[i].set_xlabel('Number of Calibration Images')
    axs[i].set_ylabel('Distortion Value')
    axs[i].set_title(f'Learning Curves for {dist_labels[i]}/Std Deviation')
    axs[i].legend()

plt.tight_layout()
plt.show()

for index, dist_set in enumerate(dist_values):
    print(f"Set {index + 1}: {dist_set}")


dist_values_array = np.array(dist_values)

means = np.mean(dist_values_array, axis=0)
stds = np.std(dist_values_array, axis=0)

##STATISTICS##
print("Means of distortion coefficients:", means)
print("Standard deviations of distortion coefficients:", stds)
mins = np.min(dist_values_array, axis=0)
maxs = np.max(dist_values_array, axis=0)
print("Minimum values of distortion coefficients:", mins)
print("Maximum values of distortion coefficients:", maxs)
"""


###LEARNING CURVE - KFOLD, TRAIN, TEST###
def collect_parameters_kfold(images, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_results = []
    test_results = []

    for train_index, test_index in kf.split(images):
        train_imgs = [images[i] for i in train_index]
        test_imgs = [images[i] for i in test_index]

        _, cameraMatrix_train, dist_train, rvecs_train, tvecs_train, _, _ = calibrate_camera(train_imgs)
        _, cameraMatrix_test, dist_test, rvecs_test, tvecs_test, _, _ = calibrate_camera(test_imgs)

        train_results.append(
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

        test_results.append(
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

    return train_results, test_results


train_results, test_results = collect_parameters_kfold(all_images)

# PLOTTING###
num_folds = range(1, len(train_results) + 1)
fx_train = [result["fx"] for result in train_results]
fy_train = [result["fy"] for result in train_results]
cx_train = [result["cx"] for result in train_results]
cy_train = [result["cy"] for result in train_results]

fx_test = [result["fx"] for result in test_results]
fy_test = [result["fy"] for result in test_results]
cx_test = [result["cx"] for result in test_results]
cy_test = [result["cy"] for result in test_results]

degree = 7
fx_poly_train = np.polyfit(num_folds, fx_train, degree)
fy_poly_train = np.polyfit(num_folds, fy_train, degree)
cx_poly_train = np.polyfit(num_folds, cx_train, degree)
cy_poly_train = np.polyfit(num_folds, cy_train, degree)

fx_poly_test = np.polyfit(num_folds, fx_test, degree)
fy_poly_test = np.polyfit(num_folds, fy_test, degree)
cx_poly_test = np.polyfit(num_folds, cx_test, degree)
cy_poly_test = np.polyfit(num_folds, cy_test, degree)

fx_poly_y_train = np.polyval(fx_poly_train, num_folds)
fy_poly_y_train = np.polyval(fy_poly_train, num_folds)
cx_poly_y_train = np.polyval(cx_poly_train, num_folds)
cy_poly_y_train = np.polyval(cy_poly_train, num_folds)

fx_poly_y_test = np.polyval(fx_poly_test, num_folds)
fy_poly_y_test = np.polyval(fy_poly_test, num_folds)
cx_poly_y_test = np.polyval(cx_poly_test, num_folds)
cy_poly_y_test = np.polyval(cy_poly_test, num_folds)

# intrinsic values
fig, axs = plt.subplots(2, 1, figsize=(14, 12))

axs[0].plot(num_folds, fx_train, "o", label="fx Train")
axs[0].plot(num_folds, fx_poly_y_train, "-", label="fx Train Poly Fit")
axs[0].plot(num_folds, fx_test, "x", label="fx Test")
axs[0].plot(num_folds, fx_poly_y_test, "--", label="fx Test Poly Fit")
axs[0].plot(num_folds, fy_train, "o", label="fy Train")
axs[0].plot(num_folds, fy_poly_y_train, "-", label="fy Train Poly Fit")
axs[0].plot(num_folds, fy_test, "x", label="fy Test")
axs[0].plot(num_folds, fy_poly_y_test, "--", label="fy Test Poly Fit")
axs[0].set_xlabel("Fold Number")
axs[0].set_ylabel("Parameter Value")
axs[0].set_title("Learning Curves for fx and fy")
axs[0].legend()


axs[1].plot(num_folds, cx_train, "o", label="cx Train")
axs[1].plot(num_folds, cx_poly_y_train, "-", label="cx Train Poly Fit")
axs[1].plot(num_folds, cx_test, "x", label="cx Test")
axs[1].plot(num_folds, cx_poly_y_test, "--", label="cx Test Poly Fit")
axs[1].plot(num_folds, cy_train, "o", label="cy Train")
axs[1].plot(num_folds, cy_poly_y_train, "-", label="cy Train Poly Fit")
axs[1].plot(num_folds, cy_test, "x", label="cy Test")
axs[1].plot(num_folds, cy_poly_y_test, "--", label="cy Test Poly Fit")
axs[1].set_xlabel("Fold Number")
axs[1].set_ylabel("Parameter Value")
axs[1].set_title("Learning Curves for cx and cy")
axs[1].legend()

plt.tight_layout()
plt.show()

# extrinsic
rvecs_train = [result["rvecs"] for result in train_results]
tvecs_train = [result["tvecs"] for result in train_results]
rvecs_test = [result["rvecs"] for result in test_results]
tvecs_test = [result["tvecs"] for result in test_results]

rvecs_poly_train = np.polyfit(num_folds, [rv[0] for rv in rvecs_train], degree)
tvecs_poly_train = np.polyfit(num_folds, [tv[0] for tv in tvecs_train], degree)

rvecs_poly_test = np.polyfit(num_folds, [rv[0] for rv in rvecs_test], degree)
tvecs_poly_test = np.polyfit(num_folds, [tv[0] for tv in tvecs_test], degree)

rvecs_poly_y_train = np.polyval(rvecs_poly_train, num_folds)
tvecs_poly_y_train = np.polyval(tvecs_poly_train, num_folds)

rvecs_poly_y_test = np.polyval(rvecs_poly_test, num_folds)
tvecs_poly_y_test = np.polyval(tvecs_poly_test, num_folds)


fig, axs = plt.subplots(2, 1, figsize=(14, 12))

axs[0].plot(num_folds, [rv[0] for rv in rvecs_train], "o", label="rvec Train")
axs[0].plot(num_folds, rvecs_poly_y_train, "-", label="rvec Train Poly Fit")
axs[0].plot(num_folds, [rv[0] for rv in rvecs_test], "x", label="rvec Test")
axs[0].plot(num_folds, rvecs_poly_y_test, "--", label="rvec Test Poly Fit")
axs[0].set_xlabel("Fold Number")
axs[0].set_ylabel("Rotation Vector Value")
axs[0].set_title("Learning Curves for Rotation Vectors")
axs[0].legend()

axs[1].plot(num_folds, [tv[0] for tv in tvecs_train], "o", label="tvec Train")
axs[1].plot(num_folds, tvecs_poly_y_train, "-", label="tvec Train Poly Fit")
axs[1].plot(num_folds, [tv[0] for tv in tvecs_test], "x", label="tvec Test")
axs[1].plot(num_folds, tvecs_poly_y_test, "--", label="tvec Test Poly Fit")
axs[1].set_xlabel("Fold Number")
axs[1].set_ylabel("Translation Vector Value")
axs[1].set_title("Learning Curves for Translation Vectors")
axs[1].legend()

plt.tight_layout()
plt.show()

dist_train = [result["dist"] for result in train_results]
dist_test = [result["dist"] for result in test_results]
dist_labels = ["k1", "k2", "p1", "p2", "k3"]
fig, axs = plt.subplots(5, 1, figsize=(14, 30))

for i in range(5):
    dist_poly_train = np.polyfit(num_folds, [d[i] for d in dist_train], degree)
    dist_poly_y_train = np.polyval(dist_poly_train, num_folds)

    dist_poly_test = np.polyfit(num_folds, [d[i] for d in dist_test], degree)
    dist_poly_y_test = np.polyval(dist_poly_test, num_folds)

    axs[i].plot(num_folds, [d[i] for d in dist_train], "o", label=dist_labels[i] + " Train")
    axs[i].plot(num_folds, dist_poly_y_train, "-", label=f"{dist_labels[i]} Train Poly Fit")
    axs[i].plot(num_folds, [d[i] for d in dist_test], "x", label=dist_labels[i] + " Test")
    axs[i].plot(num_folds, dist_poly_y_test, "--", label=f"{dist_labels[i]} Test Poly Fit")
    axs[i].set_xlabel("Fold Number")
    axs[i].set_ylabel("Distortion Value")
    axs[i].set_title(f"Learning Curves for {dist_labels[i]}")
    axs[i].legend()

plt.tight_layout()
plt.show()

for index, dist_set in enumerate(dist_train):
    print(f"Train Set {index + 1}: {dist_set}")

for index, dist_set in enumerate(dist_test):
    print(f"Test Set {index + 1}: {dist_set}")

dist_train_array = np.array(dist_train)
dist_test_array = np.array(dist_test)
means_train = np.mean(dist_train_array, axis=0)
stds_train = np.std(dist_train_array, axis=0)
mins_train = np.min(dist_train_array, axis=0)
maxs_train = np.max(dist_train_array, axis=0)

means_test = np.mean(dist_test_array, axis=0)
stds_test = np.std(dist_test_array, axis=0)
mins_test = np.min(dist_test_array, axis=0)
maxs_test = np.max(dist_test_array, axis=0)

##STATISTICS###
""" print("Means of distortion coefficients (Train):", means_train)
print("Standard deviations of distortion coefficients (Train):", stds_train)
print("Minimum values of distortion coefficients (Train):", mins_train)
print("Maximum values of distortion coefficients (Train):", maxs_train)

print("Means of distortion coefficients (Test):", means_test)
print("Standard deviations of distortion coefficients (Test):", stds_test)
print("Minimum values of distortion coefficients (Test):", mins_test)
print("Maximum values of distortion coefficients (Test):", maxs_test) """
