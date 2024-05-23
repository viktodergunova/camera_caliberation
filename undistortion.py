import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

### ANALYSIS ON UNDISTORT IMG
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

 """    plt.figure()
    plt.scatter(range(len(distances)), distances, color="red")
    plt.title("Scatter Plot of Distances")
    plt.xlabel("Index")
    plt.ylabel("Distance (pixels)")
    plt.grid(True)
    plt.show() """


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
