###Viktoria Dergunova
import glob
from datetime import datetime
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

chessboardSize = (17, 28)  # -1
frameSize = (3976, 2652)

#####CALIBERATION######
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
size_of_chessboard_squares_mm = 10
objp *= size_of_chessboard_squares_mm

objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob("test/*.jpg")  ##PATH
# images = glob.glob("links_8bit_jpg_DS_0.5/*.jpg")  ##PATH
# print(len(images))

for image in images:
    img = cv.imread(image)
    if img is None:
        print(f"Failed to load image {image}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)

    else:
        print(f"Chessboard corners not detected in {image}")
        cv.waitKey(6000)

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None
)

# WRITE XML
current_date = datetime.now().strftime("%Y-%m-%d")

fs = cv.FileStorage("calibration.xml", cv.FILE_STORAGE_WRITE)
fs.write("calibration_date", current_date)
fs.startWriteStruct("camera_matrix", cv.FileNode_MAP)
fs.write("fx", cameraMatrix[0, 0])  # focal length of camera, x (pixel unit)
fs.write("fy", cameraMatrix[1, 1])  # focal length of camera, y (pixel unit)
fs.write(
    "cx", cameraMatrix[0, 2]
)  # optical center coordinates of camera, x (pixel unit)
fs.write(
    "cy", cameraMatrix[1, 2]
)  # optical center coordinates of camera, y (pixel unit)
fs.endWriteStruct()

fs.startWriteStruct("distortion_coefficients", cv.FileNode_MAP)
fs.write("k1", dist[0, 0])  # radial distortion coefficients
fs.write("k2", dist[0, 1])  # radial distortion coefficients
fs.write(
    "k3", dist[0, 4]
)  # radial distortion coefficients, correction for severe distortion
fs.write("p1", dist[0, 2])  # tangential distortion coefficients
fs.write("p2", dist[0, 3])  # tangential distortion coefficients
fs.endWriteStruct()

fs.write("image_height", 3976)
fs.write("image_width", 2652)

fs.release()

print("Camera matrix:\n", cameraMatrix)
print("Distortion coefficients:\n", dist.flatten())


def calculate_mean_reprojection_error_per_image(
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
        # print(f"Shape of imgpoints2: {imgpoints2.shape}")
        # print(f"Shape of observed_points: {imgpoints.shape}")
        # print(f"First few errors", errors)

    mean_error = mean_error / len(objpoints)
    return mean_errors_per_image, mean_error


def calculate_reprojection_errors_alternative(
    objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
):
    # Similar to the provided function body
    total_error = 0
    total_points = 0
    error_x_components = []
    error_y_components = []
    individual_errors = []
    coordinates = []

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist
        )
        imgpoints2 = imgpoints2.reshape(-1, 2)
        observed_points = imgpoints[i].reshape(-1, 2)

        error_x = observed_points[:, 0] - imgpoints2[:, 0]
        error_y = observed_points[:, 1] - imgpoints2[:, 1]
        error_x_components.extend(error_x)
        error_y_components.extend(error_y)

        errors = np.linalg.norm(observed_points - imgpoints2, axis=1)
        individual_errors.append(errors)
        coordinates.extend(
            observed_points.tolist()
        )  # Ensure list of lists structure for coordinates

        total_error += np.sum(errors)
        total_points += len(errors)

    mean_error = total_error / total_points if total_points != 0 else 0
    coordinates = np.array(
        coordinates
    )  # Convert coordinates to a NumPy array for plotting

    return (
        coordinates,
        error_x_components,
        error_y_components,
        individual_errors,
        mean_error,
    )


####VISUALIZATION####
def plot_residual_errors(mean_errors_per_image, mean_error):
    fig, ax = plt.subplots()
    ax.plot(
        range(len(mean_errors_per_image)),
        mean_errors_per_image,
        marker="o",
        linestyle="-",
        color="b",
        label="Individual Reprojection Error for an Image",
    )
    ax.axhline(mean_error, color="r", linestyle="--", label="Overall Mean Error")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Mean Reprojection Error in Pixels")
    ax.set_title("Mean Reprojection Error per Image (OpenCV)")
    ax.legend()
    plt.savefig("residual_plot_rpe_line.png")

    plt.show()


def visualize_points_and_errors(objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist):
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist
        )
        imgpoints2 = imgpoints2.reshape(-1, 2)

        observed_points = imgpoints[i].reshape(-1, 2)

        undistorted_points = cv.undistortPoints(
            imgpoints[i], cameraMatrix, dist, P=cameraMatrix
        )
        undistorted_points = undistorted_points.reshape(-1, 2)

        plt.figure(figsize=(10, 5))

        plt.scatter(
            observed_points[:, 0],
            observed_points[:, 1],
            color="red",
            label="Observed Points",
        )
        plt.scatter(
            imgpoints2[:, 0],
            imgpoints2[:, 1],
            marker="x",
            color="blue",
            label="Projected Points",
        )
        plt.scatter(
            undistorted_points[:, 0],
            undistorted_points[:, 1],
            marker="^",
            color="green",
            label="Undistorted Points",
        )

        # Zeichne Linien zwischen den Punkten
        for op, pp, up in zip(observed_points, imgpoints2, undistorted_points):
            plt.plot(
                [op[0], pp[0]],
                [op[1], pp[1]],
                color="blue",
                linestyle="dashed",
                linewidth=0.5,
            )
            plt.plot(
                [op[0], up[0]],
                [op[1], up[1]],
                color="green",
                linestyle="dashed",
                linewidth=0.5,
            )

        plt.legend()
        plt.title(
            f"Comparison of Observed, Projected, and Undistorted Points for Image {i+1}"
        )
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.axis("equal")
        plt.show()


# CALL FUNCTIONS
opencv, mean_error = calculate_mean_reprojection_error_per_image(
    objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
)

coordinates, error_x_components, error_y_components, individual_errors, mean_error2 = (
    calculate_reprojection_errors_alternative(
        objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
    )
)
print("Mean Reprojection Error:", mean_error2)
print("OpenCV2 Mean RPE:", mean_error)
print("RMS:", ret)


plot_residual_errors(opencv, mean_error)
# visualize_points_and_errors(objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist)


""" plt.figure(figsize=(8, 6))
plt.plot(
    images, opencv, marker="o", linestyle="-", color="b"
)  # Blue line with circle markers
plt.title("Plot of Mean Reprojection Error with Different Number of Images")
plt.xlabel("Number of Images")
plt.ylabel("Mean Reprojection Error")
plt.xticks(images)
plt.grid(True)
 """


def plot_error_norm_distribution(individual_errors):
    # Flatten the list of arrays to get a single array of all error norms
    all_error_norms = np.concatenate(individual_errors)

    plt.figure(figsize=(6, 4))
    plt.hist(all_error_norms, bins=30, color="blue")
    plt.title("Histogram of Reprojection Error Magnitudes")
    plt.xlabel("RPE")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()


plot_error_norm_distribution(individual_errors)


def plot_error_heatmap(coordinates, individual_errors):
    all_coords = np.vstack(coordinates)
    all_errors = np.concatenate(individual_errors)
    # print(all_errors)

    assert (
        all_coords.ndim == 2 and all_coords.shape[1] == 2
    ), "Coordinates array must be 2D with two columns"

    plt.figure(figsize=(10, 8))
    heatmap, xedges, yedges = np.histogram2d(
        all_coords[:, 0], all_coords[:, 1], bins=100, weights=all_errors, density=True
    )
    plt.imshow(
        heatmap.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        cmap="hot",
        interpolation="nearest",
    )
    plt.colorbar(label="Error Magnitude")
    plt.title("Distribution of Reprojection Error Magnitudes over all Images")
    plt.xlabel("X Coordinate (pixel)")
    plt.ylabel("Y Coordinate (pixel)")
    plt.show()


plot_error_heatmap(coordinates, individual_errors)


def plot_error_directions(coordinates, error_x_components, error_y_components):
    coordinates = np.array(coordinates)
    error_x_components = np.array(error_x_components)
    error_y_components = np.array(error_y_components)

    if (
        coordinates.size == 0
        or error_x_components.size == 0
        or error_y_components.size == 0
    ):
        print("Input data arrays are empty.")
        return

    angles = np.arctan2(error_y_components, error_x_components)
    try:
        vor = Voronoi(coordinates)
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        voronoi_plot_2d(vor, ax=axs[0], show_vertices=False, show_points=False)
        for region, angle in zip(vor.point_region, angles):
            region_idx = vor.regions[region]
            if -1 not in region_idx:
                polygon = [vor.vertices[i] for i in region_idx]
                color = plt.cm.hsv((angle + np.pi) / (2 * np.pi))
                axs[0].fill(*zip(*polygon), color=color, edgecolor="none", linewidth=0)
        axs[0].set_title("Error Direction Diagram over all Images")
        axs[0].set_axis_off()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


plot_error_directions(coordinates, error_x_components, error_y_components)

### ANALYSIS ON UNDISTORT IMG
# TODO: Check corner detection!!! :(


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
ret, corners = cv.findChessboardCorners(
    cv.cvtColor(undistorted_img, cv.COLOR_BGR2GRAY), (17, 28)
)
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
visualize_distances_on_chessboard(undistorted_img, corners, (17, 28))
