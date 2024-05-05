###Viktoria Dergunova
# TODO: CV_CALIB_FIX_ASPECT_RATIO
import glob
from datetime import datetime
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

chessboardSize = (17, 28)  # -1
frameSize = (3976, 2652)

#####CALIBERATION######

# Termination criteria
# num of iteration: increase -> more refined results
# num of epsilon: drecrease -> tightens the convergence criteria,refines the corner locations to a finer accuracy

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
size_of_chessboard_squares_mm = 10
objp *= size_of_chessboard_squares_mm

objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob("test/*.jpg")  ##PATH
# images = glob.glob('links_8bit_jpg_DS_0.5/*.jpg') ##PATH

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # gray = cv.equalizeHist(gray)  # Equalize histogram to enhance contrast
    # gray = cv.GaussianBlur(gray, (5, 5), 0) #Reduce noise

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # links: 0.39 pixels away from corresponding observed points
    if ret:
        objpoints.append(objp)
        # subpixel
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(1000)

cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None
)
print("RMS:", ret)


# print("Distortion Coefficients Shape:", dist.shape)
# print("Distortion Coefficients:", dist)

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


def calculate_reprojection_error_components(
    objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
):
    error_x_components = []
    error_y_components = []

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist
        )
        imgpoints2 = imgpoints2.reshape(-1, 2)
        observed_points = imgpoints[i].reshape(-1, 2)

        error_x = observed_points[:, 0] - imgpoints2[:, 0]
        error_y = observed_points[:, 1] - imgpoints2[:, 1]

        error_x_components.append(error_x)
        error_y_components.append(error_y)

    return error_x_components, error_y_components


####CALCULATE RPE####


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
    # print(np.shape(objpoints)) #475, len 4
    return mean_errors_per_image, mean_error


def calculate_reprojection_errors_alternative(
    objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
):
    total_error = 0
    total_points = 0
    error_x_components = []
    error_y_components = []
    individual_errors = []

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist
        )
        imgpoints2 = imgpoints2.reshape(-1, 2)
        observed_points = imgpoints[i].reshape(-1, 2)

        # Calculate x and y error components
        error_x = observed_points[:, 0] - imgpoints2[:, 0]
        error_y = observed_points[:, 1] - imgpoints2[:, 1]
        error_x_components.extend(error_x)
        error_y_components.extend(error_y)

        # Calculate the norm for each point's error
        errors = np.linalg.norm(observed_points - imgpoints2, axis=1)

        # print(f"Shape of imgpoints2: {imgpoints2.shape}")
        # print(f"Shape of observed_points: {observed_points.shape}")
        # print(f"First few errors: {errors[:5]}")

        total_error += np.sum(errors)
        total_points += len(errors)
        individual_errors.append(errors)

    mean_error = total_error / total_points if total_points != 0 else 0

    print(individual_errors)
    return error_x_components, error_y_components, individual_errors, mean_error


####VISUALIZATION####
def plot_residual_errors(mean_errors_per_image, mean_error):
    fig, ax = plt.subplots()
    # Create the scatter plot for individual mean reprojection errors per image
    ax.scatter(
        range(len(mean_errors_per_image)),
        mean_errors_per_image,
        label="Individual Reprojection Error for an Image",
    )
    # Add a horizontal line for the overall mean error
    ax.axhline(mean_error, color="r", linestyle="-", label="Overall Mean Error")
    # Add labels and title
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Mean Reprojection Error in Pixels")
    ax.set_title("Mean Reprojection Error per Image (OpenCV)")
    # Add legend
    ax.legend()
    plt.savefig("residual_plot_rpe.png")
    # Show the plot
    plt.show()


def visualize_points_and_errors(objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist):
    for i in range(len(objpoints)):
        # Projizierte Punkte berechnen
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
visualize_points_and_errors(objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist)

opencv, mean_error = calculate_mean_reprojection_error_per_image(
    objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
)
x_components, y_components, individual_errors, mean_error2 = (
    calculate_reprojection_errors_alternative(
        objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
    )
)


print("Mean Reprojection Error:", mean_error2)
print("OpenCV2 Mean RPE:", mean_error)
print("RMS:", ret)


plot_residual_errors(opencv, mean_error)
visualize_points_and_errors(objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist)


error_x_components, error_y_components, _, _ = (
    calculate_reprojection_errors_alternative(
        objpoints, imgpoints, cameraMatrix, rvecs, tvecs, dist
    )
)

plt.figure(figsize=(8, 6))
plt.hist2d(error_x_components, error_y_components, bins=30, cmap="Blues")
plt.colorbar(label="Count")
plt.title("Histogram of Reprojection Errors")
plt.xlabel("Error X (pixels)")
plt.ylabel("Error Y (pixels)")
plt.show()

def visualize_reprojection_errors_log_scale(error_x_components, error_y_components):
    plt.figure(figsize=(10, 6))
    
    error_x = np.concatenate(error_x_components)
    error_y = np.concatenate(error_y_components)

    error_vectors = np.vstack((error_x, error_y)).T
    error_magnitude = np.linalg.norm(error_vectors, axis=1)
    error_direction = np.arctan2(error_y, error_x)

    # Apply a logarithmic scale to error magnitude; add a small constant to avoid log(0)
    error_magnitude_log = np.log10(error_magnitude + 1e-10)

    scatter = plt.scatter(error_x, error_y, c=error_direction, cmap='hsv', alpha=0.75, 
                          s=10 * (error_magnitude_log / error_magnitude_log.max()))
    plt.colorbar(scatter, label='Error Direction (Radians)')
    plt.title('Log-Scaled Reprojection Error Directions')
    plt.xlabel('Error X Component (pixels)')
    plt.ylabel('Error Y Component (pixels)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

visualize_reprojection_errors_log_scale(error_x_components, error_y_components)
