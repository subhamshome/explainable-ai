import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Saliency:
    def __init__(self, saliency_map):
        self.saliency_map = saliency_map

    def get_saliency(self, x, y):
        return self.saliency_map[y, x]

    def is_unsigned(self):
        return np.min(self.saliency_map) >= 0

    def is_signed(self):
        return not self.is_unsigned()


class Cmap:
    def __init__(self, cmap_object):
        if not isinstance(cmap_object, plt.Colormap):
            raise ValueError("cmap_object must be a matplotlib colormap")
        self.cmap_object = cmap_object


class RGBImage:
    def __init__(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError("image must be a Pillow Image object")

        self.image = image.convert("RGBA")
