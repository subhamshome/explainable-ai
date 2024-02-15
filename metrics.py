# get insertion, deletion, PCC and SSIM metrics
import numpy as np
import tensorflow as tf
import cv2 as cv
from scipy.stats import pearsonr
from utils import (
    get_index_from_image,
)

# Load the pre-trained model
model = tf.keras.models.load_model("resnet50_tl_mex.h5", compile=False)


def set_n_pixels_deletion(image_array, saliency_map, n_pixels):
    """
    Set n_pixels number of pixels to zero based on saliency, simulating deletion.

    Args:
        image_array (np.ndarray): The input image array.
        saliency_map (np.ndarray): The saliency map.
        n_pixels (int): The number of pixels to set to zero.

    Returns:
        np.ndarray: Modified image array.
        np.ndarray: Modified saliency map.
    """
    flattened_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flattened_saliency)[::-1][:n_pixels]
    flattened_saliency[sorted_indices] = 0
    modified_saliency = flattened_saliency.reshape(saliency_map.shape)

    row_indices, col_indices = np.unravel_index(sorted_indices, saliency_map.shape)
    modified_image = np.squeeze(image_array)
    modified_image[row_indices, col_indices, :] = 0
    modified_image = modified_image.reshape(1, *modified_image.shape)
    return modified_image, modified_saliency


def set_n_pixels_insertion(image_array, blurred_image, saliency_map, n_pixels):
    """
    Set n_pixels number of pixels based on saliency in the blurred image, simulating insertion.

    Args:
        image_array (np.ndarray): The input image array.
        blurred_image (np.ndarray): The blurred image array.
        saliency_map (np.ndarray): The saliency map.
        n_pixels (int): The number of pixels to set based on saliency.

    Returns:
        np.ndarray: Modified blurred image array.
        np.ndarray: Modified saliency map.
    """
    flattened_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flattened_saliency)[::-1][:n_pixels]
    flattened_saliency[sorted_indices] = 0
    modified_saliency = flattened_saliency.reshape(saliency_map.shape)

    row_indices, col_indices = np.unravel_index(sorted_indices, saliency_map.shape)
    modified_blurred_image = np.squeeze(blurred_image)
    image_array = np.squeeze(image_array)
    modified_blurred_image[row_indices, col_indices, :] = image_array[
        row_indices, col_indices, :
    ]
    modified_blurred_image = modified_blurred_image.reshape(
        1, *modified_blurred_image.shape
    )
    return modified_blurred_image, modified_saliency


def predict_scores(image, filepath):
    """
    Predict the score for a given image using the loaded model.

    Args:
        image (np.ndarray): The input image array.
        filepath (str): The file path of the image.

    Returns:
        float: The predicted score.
    """
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image).flatten()
    image = np.squeeze(image)
    index = get_index_from_image(filepath)
    score = preds[index]

    return score


def deletion(image_array, filepath, saliency_map, n_pixels):
    """
    Simulate deletion by iteratively setting pixels to zero based on saliency.

    Args:
        image_array (np.ndarray): The input image array.
        filepath (str): The file path of the image.
        saliency_map (np.ndarray): The saliency map.
        n_pixels (int): The maximum number of pixels to set to zero.

    Returns:
        float: The deletion score (AUC).
        list: Scores at each deletion step.
        list: Normalized values of n_pixels at each step.
    """
    scores = []
    image_array1 = image_array.copy()
    image_slice = np.squeeze(image_array1)[:, :, 0]
    n = 0
    score = predict_scores(image_array1, filepath)
    scores.append(score)
    while np.any(image_slice != 0):
        print("Deletion")
        image_array1, saliency_map = set_n_pixels_deletion(
            image_array1, saliency_map, n_pixels
        )
        image_slice = np.squeeze(image_array1)[:, :, 0]
        n = n + 1
        image_array1 = np.squeeze(image_array1)
        score = predict_scores(image_array1, filepath)
        scores.append(score)

        if n > n_pixels:
            score = 0
            scores.append(score)
            n = n + 1
            break

    n_values = [index / n for index in range(n + 1)]
    score_d = np.trapz(scores, n_values)

    return score_d, scores, n_values


def insertion(image_array, filepath, saliency_map, n_pixels):
    """
    Simulate insertion by iteratively setting pixels based on saliency in the blurred image.

    Args:
        image_array (np.ndarray): The input image array.
        filepath (str): The file path of the image.
        saliency_map (np.ndarray): The saliency map.
        n_pixels (int): The maximum number of pixels to set based on saliency.

    Returns:
        float: The insertion score (AUC).
        list: Scores at each insertion step.
        list: Normalized values of n_pixels at each step.
    """
    scores = []
    image_slice = np.squeeze(image_array)[:, :, 0]
    blurred_image_array = cv.GaussianBlur(image_array, (127, 127), 0)
    blurred_image_slice = blurred_image_array[:, :, 0]
    n = 0
    score = 0
    scores.append(score)
    while np.any(blurred_image_slice != image_slice):
        print("Insertion")
        blurred_image_array, saliency_map = set_n_pixels_insertion(
            image_array, blurred_image_array, saliency_map, n_pixels
        )
        blurred_image_slice = np.squeeze(blurred_image_array)[:, :, 0]
        n = n + 1
        blurred_image_array = np.squeeze(blurred_image_array)
        score = predict_scores(blurred_image_array, filepath)
        scores.append(score)

        if n > n_pixels:
            score = 1
            scores.append(score)
            n = n + 1
            break

    n_values = [index / n for index in range(n + 1)]
    score_i = np.trapz(scores, n_values)

    return score_i, scores, n_values


def calculate_pcc(gt, map):
    """Calculates the Pearson correlation coefficient (PCC) between two images.

    Args:
        gt: A numpy array representing the ground truth image.
        map: A numpy array representing the predicted image.

    Returns:
        A float representing the PCC between the two images.
    """

    gt_flattened = gt.flatten()
    map_flattened = map.flatten()
    pcc, _ = pearsonr(gt_flattened, map_flattened)

    return pcc


def calculate_sim(gt, map):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        gt (np.ndarray): A numpy array representing the ground truth image.
        map (np.ndarray): A numpy array representing the predicted image.

    Returns:
        float: The SSIM between the two images.
    """

    gt = (gt - gt.min()) / (gt.max() - gt.min())
    gt = gt / np.sum(gt)
    map = (map - map.min()) / (map.max() - map.min())
    map = map / np.sum(map)
    sim = np.sum(np.minimum(gt, map))

    return sim
