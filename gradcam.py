# gradcam.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from utils import (
    get_last_conv_layer,
    get_index_from_image,
)

model = tf.keras.models.load_model("resnet50_tl_mex.h5", compile=False)


def compute_gradients(
    model_name: str, model: Model, class_name: str, img_array: np.ndarray, filepath: str
) -> tuple:
    """
    Compute gradients and last convolutional layer output for Grad-CAM.

    Args:
        model_name (str): The name of the target model.
        grad_cam_model (Model): The Grad-CAM model.
        class_name (str): The target class name.
        img_array (np.ndarray): The input image array.

    Returns:
        tuple: A tuple containing gradients and the last convolutional layer output.
    """
    # preprocess_input = get_preprocess_input(model_name)
    # img_array = tf.expand_dims(preprocess_input(img_array), axis=0)
    model.layers[-1].activation = None

    last_layer_name = get_last_conv_layer(model_name)

    # Create a model for Grad-CAM using the specified last layer
    grad_cam_model = Model(
        model.inputs, [model.get_layer(last_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        img_array = np.expand_dims(img_array, axis=0)
        last_conv_layer_output, preds = grad_cam_model(img_array)

        # sort the predictions so that the class_index gives the correct score
        preds_sorted = tf.sort(preds, direction="DESCENDING")

        # get the class index
        index = get_index_from_image(filepath)
        # score = preds_sorted[index]

        # use the sorted predictions to get the score
        score = preds_sorted[:, index]

    gradients = tape.gradient(score, last_conv_layer_output)

    return gradients, last_conv_layer_output


def pool_gradients(gradients: tf.Tensor) -> list:
    """
    Pool gradients by performing global average pooling for each channel.

    Args:
        gradients (tf.Tensor): Gradients from the Grad-CAM computation.

    Returns:
        list: A list of pooled gradients for each channel.
    """
    return [
        tf.keras.layers.GlobalAveragePooling2D()(
            tf.expand_dims(gradients[:, :, :, channel_index], axis=-1)
        )
        for channel_index in range(gradients.shape[-1])
    ]


def calculate_weighted_activation_maps(
    last_conv_layer_output: tf.Tensor, pooled_gradients: list
) -> np.ndarray:
    """
    Calculate weighted activation maps for each channel.

    Args:
        last_conv_layer_output (tf.Tensor): Output of the last convolutional layer.
        pooled_gradients (list): List of pooled gradients for each channel.

    Returns:
        np.ndarray: Weighted activation maps.
    """
    shape = last_conv_layer_output.shape.as_list()[1:]
    weighted_maps = np.empty(shape)

    if last_conv_layer_output.shape[-1] != len(pooled_gradients):
        print("error, size mismatch")
    else:
        for i in range(len(pooled_gradients)):
            weighted_maps[:, :, i] = (
                np.squeeze(last_conv_layer_output.numpy()[:, :, :, i], axis=0)
                * pooled_gradients[i]
            )

    return weighted_maps


def relu(weighted_maps: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU (Rectified Linear Unit) activation to weighted maps.

    Args:
        weighted_maps (np.ndarray): Weighted activation maps.

    Returns:
        np.ndarray: Activation maps with ReLU applied.
    """
    weighted_maps[weighted_maps < 0] = 0

    return weighted_maps


def apply_dimension_average_pooling(weighted_maps: np.ndarray) -> np.ndarray:
    """
    Apply average pooling along the channel dimension.

    Args:
        weighted_maps (np.ndarray): Weighted activation maps.

    Returns:
        np.ndarray: Resulting activation maps after average pooling.
    """
    return np.mean(weighted_maps, axis=2)
