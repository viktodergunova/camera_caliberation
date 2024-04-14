import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

###VISUALIZATION COEFFICIENTS###

fs_read = cv.FileStorage("calibration.xml", cv.FILE_STORAGE_READ)

# read camera matrix
camera_matrix_node = fs_read.getNode("camera_matrix")
fx = camera_matrix_node.getNode("fx").real()
fy = camera_matrix_node.getNode("fy").real()
cx = camera_matrix_node.getNode("cx").real()
cy = camera_matrix_node.getNode("cy").real()
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# read distortion coefficients
distortion_coeffs_node = fs_read.getNode("distortion_coefficients")
k1 = distortion_coeffs_node.getNode("k1").real()
k2 = distortion_coeffs_node.getNode("k2").real()
k3 = distortion_coeffs_node.getNode("k3").real()
p1 = distortion_coeffs_node.getNode("p1").real()
p2 = distortion_coeffs_node.getNode("p2").real()
distortion_coeffs = np.array([k1, k2, p1, p2, k3])


coeff_keys = ['k1', 'k2', 'p1', 'p2', 'k3']

coefficients = {
    'k1': k1,
    'k2': k2,
    'k3': k3,
    'p1': p1,
    'p2': p2
}


w, h = 3976, 2652
step = 100


x, y = np.meshgrid(np.linspace(0, w, w // step), np.linspace(0, h, h // step))
pts = np.vstack((x.flatten(), y.flatten())).astype(np.float32).T

# undistorted points
undistorted_pts = cv.undistortPoints(np.expand_dims(pts, axis=1), camera_matrix, np.zeros(5), None, camera_matrix)
undistorted_pts = undistorted_pts.reshape(-1, 2)

#loop through k1,k2,k3,p1,p2
for coeff_name in coeff_keys:
    coeff_value = coefficients[coeff_name]
    
    # apply distortion
    current_distortion_coeffs = np.zeros(5)
    current_distortion_coeffs[coeff_keys.index(coeff_name)] = coeff_value
    distorted_pts = cv.undistortPoints(np.expand_dims(pts, axis=1), camera_matrix, current_distortion_coeffs, None, camera_matrix)
    distorted_pts = distorted_pts.reshape(-1, 2)
    
    # vector distortion
    dx = distorted_pts[:, 0] - undistorted_pts[:, 0]
    dy = distorted_pts[:, 1] - undistorted_pts[:, 1]
    
    # heatmap
    distances = np.sqrt(dx**2 + dy**2).reshape((h // step, w // step))
    

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(f'Distortion Coefficient: {coeff_name.upper()} ({coeff_value})')

   
    # distortion vector field plot
    scale_value = 250  # Start with this value and adjust as necessary
    Q = axs[0].quiver(undistorted_pts[:, 0], undistorted_pts[:, 1], dx, dy, color='red', scale=scale_value)
    axs[0].set_title('Distortion Vector Field')
    axs[0].set_xlabel('x pixel position')
    axs[0].set_ylabel('y pixel position')
    axs[0].invert_yaxis()

    # heatmap plot
    max_dist = np.percentile(distances, 95)  # 95th percentile as the max
    heatmap = axs[1].imshow(distances, cmap='hot', interpolation='nearest', extent=[0, w, h, 0], vmax=max_dist)
    cbar = plt.colorbar(heatmap, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label('Distortion Magnitude')
    axs[1].set_title('Heatmap of Distortion Magnitude')
    axs[1].set_xlabel ('x pixel position')
    axs[1].set_ylabel('y pixel position')

    # scatter plot
    axs[2].scatter(undistorted_pts[:, 0], undistorted_pts[:, 1], c='blue', label='Undistorted Grid', s=10, alpha=0.6)
    axs[2].scatter(distorted_pts[:, 0], distorted_pts[:, 1], c='red', label='Distorted Grid', s=10, alpha=0.6)
    axs[2].set_title('Distorted Grid Points Over Undistorted Grid Points')
    axs[2].set_xlabel ('x pixel position')
    axs[2].set_ylabel('y pixel position')
    axs[2].legend()
    axs[2].invert_yaxis()

    plt.savefig(f'{coeff_name}_distortion_plots.png', bbox_inches='tight')
    plt.show()