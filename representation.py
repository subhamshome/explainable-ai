import numpy as np
from PIL import Image
from typing import Union
import matplotlib.pyplot as plt
from objects import Saliency, RGBImage, Cmap


def represent_heatmap(saliency: Saliency, cmap: Union[None, Cmap]) -> RGBImage:
    """
    Generates a heatmap of the given saliency map using the given colormap.

    Args:
        saliency: A 2D numpy array representing the saliency map.
        cmap: A matplotlib colormap object.

    Returns:
        An RGBImage representing the heatmap.
    """
    map_array = np.array(saliency)
    colormap = plt.get_cmap(cmap)
    colored_map_array = (colormap(map_array) * 255).astype(np.uint8)
    colored_map = Image.fromarray(colored_map_array)
    return colored_map


def represent_heatmap_overlaid(
    saliency: Saliency, image: RGBImage, alpha: float, cmap: Union[None, Cmap]
) -> RGBImage:
    """
    Overlays the given saliency map on the given image using the given colormap.

    Args:
        saliency: A 2D numpy array representing the saliency map.
        image: An RGBImage representing the image.
        cmap: A matplotlib colormap object.

    Returns:
        An RGBImage representing the overlaid image.
    """
    map = represent_heatmap(saliency, cmap)
    map = map.resize(image.size)
    map = map.convert("RGBA")
    image = image.convert("RGBA")
    blended_image = Image.blend(image, map, alpha)

    return blended_image
