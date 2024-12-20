import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# --- Step 1: Preprocessing ---

def preprocess_image(image_path):
    """
    Load an image, convert to grayscale, and normalize it.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found: " + image_path)
    image = cv2.resize(image, (512, 512))  # Resize to a fixed size
    image = image / 255.0  # Normalize pixel values
    return image

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def find_argmax_points(x, y):
    """
    Finds the x-values where the y-values have local maxima.

    Args:
        x (np.ndarray): Array of x-values.
        y (np.ndarray): Array of y-values corresponding to x.

    Returns:
        np.ndarray: Array of x-values at which local maxima occur.
    """
    maxima_indices = (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1  # Indices of local maxima
    separ_lines = [x[idx] for idx in maxima_indices]

    return separ_lines

def split_into_lines(image):
    """
    Split a preprocessed image into individual text lines using sliding box fashion.
    """
    height, width = image.shape
    box_height = 5  # Height of the sliding box
    step = 2  # Step size for the sliding window

    all_steps_inf_content = []
    positions = []

    for pos in range(0, height - box_height + 1, step):
        image_seg = image[pos :pos + box_height, :]
        sum_info_content = np.sum(image_seg)
        all_steps_inf_content.append(sum_info_content)
        positions.append(pos)

    np_positions = np.array(positions)
    np_all_steps_inf_content = np.array(all_steps_inf_content)

    # smooth curve
    sm_np_all_steps_inf_content = savgol_filter(np_all_steps_inf_content, window_length=11, polyorder=3)

    # determine local maxima
    maxima_indices = (np.diff(np.sign(np.diff(sm_np_all_steps_inf_content))) < 0).nonzero()[0] + 1  # Indices of local maxima
    separ_lines = [np_positions[idx] for idx in maxima_indices]

    # use local maxima posistions as speration lines

    plt.plot(positions, all_steps_inf_content)
    plt.plot(positions, sm_np_all_steps_inf_content)
    plt.show()

    lines = []
    for idx, pos in enumerate(separ_lines):
        if idx < len(separ_lines) - 1:
            line = image[pos :separ_lines[idx+1], :]
            lines.append(line)


    return lines