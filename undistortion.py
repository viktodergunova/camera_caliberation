import glob
import xml.etree.ElementTree as ET

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

""" ### ANALYSIS ON UNDISTORT IMG
# TODO: Check corner detection!!! :(

def load_calibration_parameters(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    cameraMatrix = np.array([[float(root.find('camera_matrix/fx').text), 0, float(root.find('camera_matrix/cx').text)],
                              [0, float(root.find('camera_matrix/fy').text), float(root.find('camera_matrix/cy').text)],
                              [0, 0, 1]])

    dis = np.array([float(root.find('distortion_coefficients/k1').text),
                                        float(root.find('distortion_coefficients/k2').text),
                                        float(root.find('distortion_coefficients/p1').text),
                                        float(root.find('distortion_coefficients/p2').text),
                                        float(root.find('distortion_coefficients/k3').text)])

    image_width = int(root.find('image_width').text)
    image_height = int(root.find('image_height').text)

    return cameraMatrix, dis, (image_width, image_height)

cameraMatrix, dist, _ = load_calibration_parameters("calibration.xml")

# alpha=1 pixels are retained with some extra black images
def undistort_image(image_path, cameraMatrix, dist):
    img = cv.imread(image_path)
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
        cameraMatrix, dist, (w, h), 1, (w, h)
    )
    undistorted_img = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    return undistorted_img, newCameraMatrix


def calculate_distances(corners, chessboard_size):
    distances = []
    # horizontal distances
    for row in range(chessboard_size[1]):
        for col in range(chessboard_size[0] - 1):
            idx = row * chessboard_size[0] + col
            dist = np.linalg.norm(corners[idx] - corners[idx + 1])
            distances.append(dist)
    # vertical distances
    for col in range(chessboard_size[0]):
        for row in range(chessboard_size[1] - 1):
            idx = row * chessboard_size[0] + col
            dist = np.linalg.norm(corners[idx] - corners[idx + chessboard_size[0]])
            distances.append(dist)
    return distances


def visualize_corners(image, corners):
    img = image.copy()
    if corners is not None:
        for i in range(corners.shape[0]):
            cv.circle(img, tuple(corners[i, 0].astype(int)), 5, (0, 255, 0), -1)
    cv.imshow("Chessboard Corners", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def plot_distances(distances):
    plt.figure()
    plt.hist(distances, bins=20, color="blue", alpha=0.7)
    plt.title("Distribution of Distances Between Chessboard Corners")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.scatter(range(len(distances)), distances, color="red")
    plt.title("Scatter Plot of Distances")
    plt.xlabel("Index")
    plt.ylabel("Distance (pixels)")
    plt.grid(True)
    plt.show()


def visualize_distances_on_chessboard(image, corners, chessboard_size):
    img = image.copy()
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range(chessboard_size[1]):
        for j in range(chessboard_size[0] - 1):
            start_point = tuple(int(x) for x in corners[i * chessboard_size[0] + j][0])
            end_point = tuple(
                int(x) for x in corners[i * chessboard_size[0] + j + 1][0]
            )
            distance = np.linalg.norm(np.array(start_point) - np.array(end_point))
            midpoint = (
                (start_point[0] + end_point[0]) // 2,
                (start_point[1] + end_point[1]) // 2,
            )
            cv.line(img, start_point, end_point, (0, 255, 0), 2)
            cv.putText(img, f"{distance:.2f}", midpoint, font, 0.5, (255, 255, 255), 1)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Visualized Distances on Chessboard")
    plt.axis("off")
    plt.show()


undistorted_img, newCameraMatrix = undistort_image(
    "links_8bit_jpg_DS_0.5/DSC00029_DS_0.5.jpg", cameraMatrix, dist
)
gray = cv.cvtColor(undistorted_img, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(
    cv.cvtColor(undistorted_img, cv.COLOR_BGR2GRAY), (17, 28)
)
#subpixel
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

if corners is not None:
    distances = calculate_distances(corners2, (17, 28))
    print("Mean distance:", np.mean(distances))
    print("Standard deviation:", np.std(distances))
    plot_distances(distances)
else:
    print("Chessboard corners could not be detected.")

# visualize_corners(undistorted_img, corners2)
visualize_distances_on_chessboard(undistorted_img, corners2, (17, 28))
visualize_corners(undistorted_img, corners2)
 """

### DISTORTION ANALYSIS, RECALIBEREATE, FITTED LINES
# TODO: check stdDev for vertical lines
frameSize = (3976, 2652)
size_of_chessboard_squares_mm = 10
chessboardSize = (17, 28)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm


def calibrate_camera(images, from_images=False):
    objpoints = []
    imgpoints = []
    for image in images:
        if from_images:
            img = image
        else:
            img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    return ret, cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints


def undistort_image(image, cameraMatrix, dist):
    h, w = image.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))
    undistorted_img = cv.undistort(image, cameraMatrix, dist, None, newCameraMatrix)
    return undistorted_img, newCameraMatrix


def calculate_reprojection_error(objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs):
    mean_error = 0
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
        errors.append(error)
    return mean_error / len(objpoints), errors


def measure_checkerboard_straightness(corners, chessboardSize):
    horizontal_angles = []
    vertical_angles = []

    # horizontal lines
    for i in range(chessboardSize[1]):
        for j in range(chessboardSize[0] - 1):
            pt1 = corners[i * chessboardSize[0] + j][0]
            pt2 = corners[i * chessboardSize[0] + j + 1][0]
            angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180 / np.pi
            horizontal_angles.append(angle)

    # vertical lines
    for i in range(chessboardSize[0]):
        for j in range(chessboardSize[1] - 1):
            pt1 = corners[j * chessboardSize[0] + i][0]
            pt2 = corners[(j + 1) * chessboardSize[0] + i][0]
            angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180 / np.pi
            vertical_angles.append(angle)

    return np.std(horizontal_angles), np.std(vertical_angles)


def visualize_checkerboard_lines(image, corners, chessboard_size):
    img = image.copy()
    if corners is not None:
        for i in range(corners.shape[0]):
            cv.circle(img, tuple(corners[i, 0].astype(int)), 5, (0, 255, 0), -1)

        # Draw lines
        for row in range(chessboard_size[1]):
            for col in range(chessboard_size[0] - 1):
                idx = row * chessboard_size[0] + col
                start = tuple(corners[idx, 0].astype(int))
                end = tuple(corners[idx + 1, 0].astype(int))
                cv.line(img, start, end, (255, 0, 0), 2)

        for col in range(chessboard_size[0]):
            for row in range(chessboard_size[1] - 1):
                idx = row * chessboard_size[0] + col
                start = tuple(corners[idx, 0].astype(int))
                end = tuple(corners[idx + chessboard_size[0], 0].astype(int))
                cv.line(img, start, end, (0, 0, 255), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Checkerboard Lines")
    plt.axis("off")
    plt.show()


all_images = glob.glob("./data/rechts_8bit_jpg_DS_0.5/*.jpg")
#all_images = glob.glob("./data/rechts_8bit_jpg/*.jpg")

# checkerboard straightness before any calibration
img = cv.imread(all_images[70])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
if ret:
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    h_std, v_std = measure_checkerboard_straightness(corners2, chessboardSize)
    print(f"Before Calibration - Horizontal StdDev: {h_std:.4f}, Vertical StdDev: {v_std:.4f}")
    visualize_checkerboard_lines(img, corners2, chessboardSize)
else:
    print("Checkerboard corners could not be detected.")

# initial calibration
ret, cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints = calibrate_camera(all_images)
initial_rpe, initial_rpe_errors = calculate_reprojection_error(objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs)

# checkerboard straightness after initial calibration
if ret:
    imgpoints = [cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) for corners in imgpoints]
    horizontal_std, vertical_std = measure_checkerboard_straightness(imgpoints[0], chessboardSize)
    print(f"Initial Calibration - Horizontal StdDev: {horizontal_std:.4f}, Vertical StdDev: {vertical_std:.4f}")

print(f"Initial Calibration Parameters:\nCamera Matrix:\n{cameraMatrix}\nDistortion Coefficients:\n{dist}\n")

# iterative undistortion and recalibration
num_iterations = 10
dist_coeffs_all = [dist.flatten()]
rpe_all = [initial_rpe]
rpe_errors_all = [initial_rpe_errors]
fx_all, fy_all, cx_all, cy_all = [cameraMatrix[0, 0]], [cameraMatrix[1, 1]], [cameraMatrix[0, 2]], [cameraMatrix[1, 2]]
rvecs_all, tvecs_all = [np.mean(rvecs, axis=0).flatten()], [np.mean(tvecs, axis=0).flatten()]
horizontal_std_all, vertical_std_all = [horizontal_std], [vertical_std]

undistorted_images = [cv.imread(image) for image in all_images]

for iteration in range(1, num_iterations + 1):
    undistorted_images = [undistort_image(img, cameraMatrix, dist)[0] for img in undistorted_images]

    ret, cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints = calibrate_camera(
        undistorted_images, from_images=True
    )
    rpe, rpe_errors = calculate_reprojection_error(objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs)

    dist_coeffs_all.append(dist.flatten())
    rpe_all.append(rpe)
    rpe_errors_all.append(rpe_errors)
    fx_all.append(cameraMatrix[0, 0])
    fy_all.append(cameraMatrix[1, 1])
    cx_all.append(cameraMatrix[0, 2])
    cy_all.append(cameraMatrix[1, 2])
    rvecs_all.append(np.mean(rvecs, axis=0).flatten())
    tvecs_all.append(np.mean(tvecs, axis=0).flatten())

    horizontal_std_iter = []
    vertical_std_iter = []
    for corners in imgpoints:
        if corners is not None:
            h_std, v_std = measure_checkerboard_straightness(corners, chessboardSize)
            horizontal_std_iter.append(h_std)
            vertical_std_iter.append(v_std)
    horizontal_std_all.append(np.mean(horizontal_std_iter))
    vertical_std_all.append(np.mean(vertical_std_iter))

    print(f"Re-Calibration Parameters (Iteration {iteration}):")
    print(f"Camera Matrix:\n{cameraMatrix}\nDistortion Coefficients:\n{dist}\n")


rvecs_all = np.vstack(rvecs_all)
tvecs_all = np.vstack(tvecs_all)

print(f"Shape of rvecs_all: {rvecs_all.shape}")
print(f"Shape of tvecs_all: {tvecs_all.shape}")

# StdDev Checkerboard Straightness
print(
    f"Initial Calibration - Horizontal StdDev: {horizontal_std_all[0]:.4f}, Vertical StdDev: {vertical_std_all[0]:.4f}"
)
for iteration in range(1, num_iterations + 1):
    print(
        f"Iteration {iteration} - Horizontal StdDev: {horizontal_std_all[iteration]:.4f}, Vertical StdDev: {vertical_std_all[iteration]:.4f}"
    )


labels = ["k1", "k2", "p1", "p2", "k3"]
x = np.arange(len(labels))
width = 0.1

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(num_iterations + 1):
    ax.bar(x + i * width, dist_coeffs_all[i], width, label=f"Iteration {i}")

ax.set_ylabel("Values")
ax.set_title("Distortion Coefficients Across Iterations")
ax.set_xticks(x + width * (num_iterations / 2))
ax.set_xticklabels(labels)
ax.legend()
plt.show()


fig, ax = plt.subplots(figsize=(12, 6))
width = 0.1
intrinsic_labels = ["fx", "fy", "cx", "cy"]
iterations = np.arange(num_iterations + 1)

ax.bar(iterations - 1.5 * width, fx_all, width, label="fx")
ax.bar(iterations - 0.5 * width, fy_all, width, label="fy")
ax.bar(iterations + 0.5 * width, cx_all, width, label="cx")
ax.bar(iterations + 1.5 * width, cy_all, width, label="cy")

ax.set_ylabel("Values")
ax.set_title("Intrinsic Parameters Across Iterations")
ax.set_xticks(iterations)
ax.set_xticklabels([f"Iteration {i}" for i in range(num_iterations + 1)])
ax.legend()
plt.show()

# Plot Extrinsic Parameters
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.1
extrinsic_labels = ["rvec1", "rvec2", "rvec3"]
extrinsic_params = [rvecs_all[:, 0], rvecs_all[:, 1], rvecs_all[:, 2]]

ax.bar(iterations - 1.5 * width, rvecs_all[:, 0], width, label="rvec1")
ax.bar(iterations - 0.5 * width, rvecs_all[:, 1], width, label="rvec2")
ax.bar(iterations + 0.5 * width, rvecs_all[:, 2], width, label="rvec3")

ax.set_ylabel("Values")
ax.set_title("Rotation Vectors Across Iterations")
ax.set_xticks(iterations)
ax.set_xticklabels([f"Iteration {i}" for i in range(num_iterations + 1)])
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
width = 0.1
extrinsic_labels = ["tvec1", "tvec2", "tvec3"]
extrinsic_params = [tvecs_all[:, 0], tvecs_all[:, 1], tvecs_all[:, 2]]

ax.bar(iterations - 1.5 * width, tvecs_all[:, 0], width, label="tvec1")
ax.bar(iterations - 0.5 * width, tvecs_all[:, 1], width, label="tvec2")
ax.bar(iterations + 0.5 * width, tvecs_all[:, 2], width, label="tvec3")

ax.set_ylabel("Values")
ax.set_title("Translation Vectors Across Iterations")
ax.set_xticks(iterations)
ax.set_xticklabels([f"Iteration {i}" for i in range(num_iterations + 1)])
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(num_iterations + 1), rpe_all, "bo-", label="RPE")
ax.set_xlabel("Iteration")
ax.set_ylabel("Reprojection Error")
ax.set_title("Reprojection Error Across Iterations")
ax.legend()
plt.show()

# checkerboard lines after initial undistortion
ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
if ret:
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    undistorted_img, _ = undistort_image(img, cameraMatrix, dist)
    visualize_checkerboard_lines(undistorted_img, corners2, chessboardSize)
else:
    print("Checkerboard corners could not be detected.")
