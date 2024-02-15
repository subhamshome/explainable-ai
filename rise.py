# Imports
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Tuple
import matplotlib.pyplot as plt
from skimage.transform import resize

from objects import RGBImage
from representation import represent_heatmap, represent_heatmap_overlaid

from utils import get_index_from_image


# Load the pre-trained model
model = tf.keras.models.load_model("resnet50_tl_mex.h5", compile=False)


def image_array_creator(
    original_image: RGBImage, image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Convert an original image to a NumPy array after resizing.

    Parameters:
    original_image: PIL.Image.Image
        A PIL Image representing the original image.
    image_size: Tuple[int, int]
        A tuple specifying the target dimensions (width, height) for resizing the image.

    Returns:
    np.ndarray
        A NumPy array representing the resized and converted image.
    """

    resized_image = original_image.resize(image_size)
    image_array = tf.keras.preprocessing.image.img_to_array(resized_image)

    return image_array


# Generate the masks
def generate_masks(
    n_masks: int, image_array: np.ndarray, probability: float, mask_dim: int
) -> np.ndarray:
    """
    Generate a set of binary masks for a given image with a specified probability of activation.

    Parameters:
        n_masks (int): The number of masks to generate.
        image_array (np.ndarray): The input image for which masks are generated.
        probability (float): The probability of activation for each pixel in the mask.
        mask_dim (int): The dimensions (height and width) of each mask.

    Returns:
        np.ndarray: A 3D numpy array containing the generated masks.
    """
    # Get the image dimensions
    h_image, w_image, _ = image_array.shape
    masks = np.empty((n_masks, h_image, w_image))

    # Create the mask dimensions
    h_mask, w_mask = mask_dim, mask_dim

    # Generate masks
    for i in range(n_masks):
        grid = (np.random.rand(1, h_mask, w_mask) < probability).astype("float32")

        # Mask generation algorithm
        C_H, C_W = np.ceil(h_image / h_mask), np.ceil(w_image / w_mask)
        h_new_mask, w_new_mask = (h_mask + 1) * C_H, (w_mask + 1) * C_W

        x, y = np.random.randint(0, C_H), np.random.randint(0, C_W)

        masks[i, :, :] = resize(
            grid[0],
            (h_new_mask, w_new_mask),
            order=1,
            mode="reflect",
            anti_aliasing=False,
        )[x : x + h_image, y : y + w_image]

    return masks


# Compute the perturbed images
def calculate_perturbed_images(image_array: np.ndarray, masks: list) -> list:
    """
    Calculate perturbed images by element-wise multiplication of masks with the input image.

    Parameters:
        image_array (np.ndarray): The original input image.
        masks (list): A list of binary masks to be applied to the image.

    Returns:
        list: A list of perturbed images, each obtained by element-wise multiplication of a mask with the input image.
    """
    perturbed_images = []
    for mask in masks:
        # Increase dimensionality
        mask = mask[..., None].repeat(3, axis=2)

        # Create the perturbations
        perturbed_image = image_array * mask
        perturbed_images.append(perturbed_image)

    return perturbed_images


# Compute prediction scores for perturbed images
def compute_predictions(
    perturbed_images: list, n_classes: int, class_label: str, filepath: str
) -> list:
    """
    Compute prediction scores for a list of perturbed images using a pre-trained model.

    Parameters:
        perturbed_images (list): A list of perturbed images for which predictions are to be computed.
        n_classes (int): Number of classes.
        class_label (str): The label for the class.

    Returns:
        list: A list of prediction scores corresponding to each perturbed image.
    """
    prediction_scores = []
    for perturbed_image in perturbed_images:
        perturbed_image = np.expand_dims(perturbed_image, axis=0)
        predictions = model.predict(perturbed_image).flatten()
        index = get_index_from_image(filepath)
        score = predictions[index]
        prediction_scores.append(score)

    return prediction_scores


def calculate_saliency_map(scores: list, masks: list) -> np.ndarray:
    """
    Calculate a saliency map based on a list of scores and masks.

    Parameters:
    scores (list): A list of scores corresponding to the importance of each mask.
    masks (list): A list of 2D arrays representing masks that highlight regions of interest.

    Returns:
    np.ndarray: A 2D numpy array representing the saliency map. It indicates the importance
               of different regions in the input masks, with higher values denoting more salient areas.
    """
    sum_of_scores = np.sum(scores)
    saliency_map = np.zeros(masks[0].shape, dtype=np.float64)
    for i, mask_i in enumerate(masks):
        score_i = scores[i]
        saliency_map += score_i * mask_i
    saliency_map /= sum_of_scores

    return saliency_map


def display_images(
    original_image: np.ndarray,
    image_array: np.ndarray,
    saliency_map: np.ndarray,
    colormap: str,
    alpha: float,
) -> Tuple[RGBImage, RGBImage]:
    """
    Display images by overlaying a saliency map on an original image using a specified colormap.

    Parameters:
    original_image (np.ndarray): A numpy array representing the original image.
    image_array (np.ndarray): A numpy array representing an image to overlay the saliency map on.
    saliency_map (np.ndarray): A numpy array representing the saliency map to overlay.
    colormap (str): A string specifying the colormap to use for representing the heatmap.
    alpha (float): A blending parameter to control the opacity of the saliency map overlay.

    Returns:
    Tuple[RGBImage, RGBImage]: A tuple of two PIL Image objects:
        1. The first image represents the saliency map as a heatmap overlaid on the original image.
        2. The second image is a resized version of the heatmap image.

    """
    saliency = Image.fromarray(saliency_map)
    cmap = colormap
    heatmap = represent_heatmap(saliency, cmap)

    # Resize heatmap to original image size
    resized_heatmap = heatmap.resize(original_image.size)

    # Normalize and display the original image
    image_array = image_array / 255.0
    image_array = (image_array * 255).astype(np.uint8)
    rgb_image = Image.fromarray(image_array)

    # Generate a blended heatmap image and visualize it
    blended_heatmap = represent_heatmap_overlaid(saliency, rgb_image, alpha, cmap)
    resized_blended_heatmap = blended_heatmap.resize(original_image.size)

    return resized_heatmap, resized_blended_heatmap
