import numpy as np
import tensorflow as tf
from PIL import Image
from lime import lime_image
import matplotlib.pyplot as plt

from utils import (
    get_model,
)

# Load the pre-trained model
model = tf.keras.models.load_model("resnet50_tl_mex.h5", compile=False)


def preprocess_img_array(model_name, img_array):
    """
    Preprocess the image array for model input.

    Parameters:
    - model_name: Name of the model (not used in this function)
    - img_array: Input image array

    Returns:
    - Preprocessed image array
    """
    img_array = np.array(tf.expand_dims(img_array, axis=0))
    return img_array


def make_prediction(img_array, model_name):
    """
    Make predictions using the loaded model.

    Parameters:
    - img_array: Input image array
    - model_name: Name of the model (not used in this function)

    Returns:
    - Predictions, Predicted index, and Pred index (redundant, returns the predicted index)
    """
    preds = model.predict(img_array).flatten()
    pred_index = np.argmax(preds)

    return preds, pred_index, pred_index


def get_lime_explanation(
    img_array,
    pred_index,
    top_labels,
    hide_color,
    num_lime_features,
    num_samples,
    model_name,
):
    """
    Generate Lime explanation for the given image.

    Parameters:
    - img_array: Input image array
    - pred_index: Predicted index
    - top_labels: Top labels to consider
    - hide_color: Color to hide (not used in this function)
    - num_lime_features: Number of Lime features
    - num_samples: Number of Lime samples
    - model_name: Name of the model (not used in this function)

    Returns:
    - Lime explanation object
    """
    _, model = get_model(model_name)
    explainer = lime_image.LimeImageExplainer(random_state=0)  # for reproductibility

    explanation = explainer.explain_instance(
        img_array,
        model.predict,
        top_labels=top_labels,
        labels=(pred_index,),
        hide_color=hide_color,
        num_features=num_lime_features,
        num_samples=num_samples,
        random_seed=0,
    )

    return explanation


def explain_with_lime(
    img_array,
    top_labels,
    hide_color,
    num_lime_features,
    num_samples,
    model_name,
):
    """
    Generate and visualize Lime explanation for the given image.

    Parameters:
    - img_array: Input image array
    - top_labels: Top labels to consider
    - hide_color: Color to hide (not used in this function)
    - num_lime_features: Number of Lime features
    - num_samples: Number of Lime samples
    - model_name: Name of the model (not used in this function)

    Returns:
    - Lime explanation heatmap (PIL Image)
    """
    preds = model.predict(img_array).flatten()
    pred_index = np.argmax(preds)
    print(pred_index)

    explanation = get_lime_explanation(
        img_array[0],
        pred_index,
        top_labels,
        hide_color,
        num_lime_features,
        num_samples,
        model_name,
    )

    ind = explanation.top_labels[0]
    map = np.vectorize(explanation.local_exp[ind].get)(explanation.segments)
    map = np.nan_to_num(map, nan=0)
    map = np.array(map)
    norm = plt.Normalize(vmin=map.min(), vmax=map.max())
    heatmap = Image.fromarray((norm(map) * 255).astype(np.uint8))

    return heatmap
