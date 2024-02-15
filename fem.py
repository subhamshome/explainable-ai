import numpy as np
import tensorflow as tf

from utils import (
    get_last_conv_layer,
    get_model,
)

model = tf.keras.models.load_model("resnet50_tl_mex.h5", compile=False)


def expand_flat_values_to_activation_shape(
    values: np.ndarray, W_layer: int, H_layer: int
) -> np.ndarray:
    """
    Expand a flat array of values to match the shape of an activation map.

    Args:
        values (numpy.ndarray): A 1D array of values to be expanded.
        W_layer (int): The width of the activation map.
        H_layer (int): The height of the activation map.

    Returns:
        numpy.ndarray: An expanded array that matches the shape of the activation map.

    """
    if False:
        # Initial implementation in original FEM paper
        expanded = np.expand_dims(values, axis=1)
        expanded = np.kron(expanded, np.ones((W_layer, 1, H_layer)))
        expanded = np.transpose(expanded, axes=[0, 2, 1])
    else:
        # Simplified implementation
        expanded = values.reshape((1, 1, -1)) * np.ones((W_layer, H_layer, len(values)))
    return expanded


def compute_binary_maps(feature_map, sigma=None):
    batch_size, W_layer, H_layer, N_channels = feature_map.shape
    thresholded_tensor = np.zeros((batch_size, W_layer, H_layer, N_channels))

    if sigma is None:
        feature_sigma = 2
    else:
        feature_sigma = sigma

    for B in range(batch_size):
        # Get the activation value of the current sample
        activation = feature_map[B, :, :, :]

        # Calculate its mean and its std per channel
        mean_activation_per_channel = tf.reduce_mean(activation, axis=[0, 1])

        std_activation_per_channel = tf.math.reduce_std(activation, axis=(0, 1))

        # mean_activation_per_channel = activation.mean(axis=(0,1))
        # std_activation_per_channel = activation.std(axis=(0,1))

        assert len(mean_activation_per_channel) == N_channels
        assert len(std_activation_per_channel) == N_channels

        # Transform the mean in the same shape than the activation maps
        """
        mean_activation_expanded = self.expand_flat_values_to_activation_shape(mean_activation_per_channel, W_layer,H_layer)
        
        # Transform the std in the same shape than the activation maps
        std_activation_expanded = self.expand_flat_values_to_activation_shape(std_activation_per_channel, W_layer,H_layer)
        """

        mean_activation_expanded = tf.reshape(
            mean_activation_per_channel, (1, 1, -1)
        ) * np.ones((W_layer, H_layer, len(mean_activation_per_channel)))

        # Transform the std in the same shape than the activation maps
        std_activation_expanded = tf.reshape(
            std_activation_per_channel, (1, 1, -1)
        ) * np.ones((W_layer, H_layer, len(std_activation_per_channel)))

        # Build the binary map
        thresholded_tensor[B, :, :, :] = tf.cast(
            (
                activation
                > (mean_activation_expanded + feature_sigma * std_activation_expanded)
            ),
            dtype=tf.int32,
        )

    return thresholded_tensor


def aggregate_binary_maps(
    binary_feature_map: np.ndarray, original_feature_map: np.ndarray
) -> np.ndarray:
    """
    Aggregate binary feature maps based on original feature maps and return normalized results.

    Args:
        binary_feature_map (numpy.ndarray): A 4D binary feature map with dimensions
            (batch_size, W_layer, H_layer, N_channels), where values are 1 or 0.
        original_feature_map (numpy.ndarray): A 4D original feature map with dimensions
            (batch_size, W_layer, H_layer, N_channels).

    Returns:
        numpy.ndarray: An aggregated feature map based on the binary feature map and the
        original feature map, with the same shape as the original feature map. The values
        are normalized to have a maximum value of 1, or the original feature map if
        the maximum value is 0.
    """
    # This weigths the binary map based on original feature map
    batch_size, W_layer, H_layer, N_channels = original_feature_map.shape

    original_feature_map = original_feature_map[0]
    binary_feature_map = binary_feature_map[0]

    # Get the weights
    channel_weights = np.mean(
        original_feature_map, axis=(0, 1)
    )  # Take means for each channel-values
    if False:
        # Original paper implementation
        expanded_weights = np.kron(
            np.ones((binary_feature_map.shape[0], binary_feature_map.shape[1], 1)),
            channel_weights,
        )
    else:
        # Simplified version
        expanded_weights = expand_flat_values_to_activation_shape(
            channel_weights, W_layer, H_layer
        )

    # Appcompute_binary_mapsly the weights on each binary feature map
    expanded_feat_map = np.multiply(expanded_weights, binary_feature_map)

    # Aggregate the feature map of each channel
    feat_map = np.sum(expanded_feat_map, axis=2)

    # Normalize the feature map
    if np.max(feat_map) == 0:
        return feat_map
    feat_map = feat_map / np.max(feat_map)
    return feat_map


def compute_fem(img_array: np.ndarray, model_name: str) -> np.ndarray:
    """
    Compute Feature Extraction Maps (FEM) based on an image array and a specified model.

    Args:
        img_array (numpy.ndarray): A 3D or 4D array representing the input image or batch
            of images. If it's 4D and the last dimension has 4 channels, the fourth channel
            is ignored.
        model_name (str): The name of the target model to be used for feature extraction.

    Returns:
        numpy.ndarray: A Feature Extraction Map (FEM) representing saliency, obtained by
        processing the input image(s) with the specified model. The shape and content of
        the FEM depend on the model and input.
    """
    model.layers[-1].activation = None
    last_conv_layer_name = get_last_conv_layer(model_name)
    fem = tf.keras.models.Model(
        inputs=model.input, outputs=model.get_layer(last_conv_layer_name).output
    )
    img_array = np.expand_dims(img_array, axis=0)

    feature_map = fem(img_array)
    # checks for channels of the image and converts to RGB if not already
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    # preprocess_input = get_preprocess_input(model_name)

    # increase dimensionality
    # img_array = np.expand_dims(preprocess_input(img_array), axis=0)

    # last_conv_layer_name = get_last_conv_layer(model_name)
    # _, model = get_model(model_name)

    # a line to remove the keract warning
    # model.compile(loss="categorical_crossentropy", optimizer="adam")

    # activations = keract.get_activations(model, img_array, auto_compile=True)
    # for k, v in activations.items():
    #     if k == last_conv_layer_name:
    #         feature_map = v
    binary_feature_map = compute_binary_maps(feature_map)
    saliency = aggregate_binary_maps(binary_feature_map, feature_map)
    return saliency


def predict(model_name: str, img_array: np.ndarray) -> None:
    """
    Make predictions using a specified model on an image array.

    Args:
        model_name (str): The name of the target model for making predictions.
        img_array (numpy.ndarray): A 3D or 4D array representing the input image or batch
            of images. If it's 4D and the last dimension has 4 channels, the fourth channel
            is ignored.

    Returns:
        None

    This function makes predictions using the specified model on the provided image(s)
    and prints the top prediction results. The predictions depend on the model and input.
    """

    # checks for channels of the image and converts to RGB if not already
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # increase dimensionality
    img_array = np.expand_dims(img_array, axis=0)

    # perform predictions
    _, model = get_model(model_name)
    preds = model.predict(img_array)
