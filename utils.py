import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import Model
from matplotlib.gridspec import GridSpec


def get_last_conv_layer(model_name: str) -> str:
    """
    Get the name of the last convolutional layer for a given model.

    Args:
        model_name (str): The name of the target model.

    Returns:
        str: The name of the last convolutional layer for the specified model.
    """
    last_conv_layer_names = {
        "Xception": "block14_sepconv2_act",
        "ResNet": "conv5_block3_out",
    }

    last_conv_layer_name = last_conv_layer_names.get(model_name)

    return last_conv_layer_name


def get_model(model_name: str) -> Model:
    """
    Get the pre-trained model of a specified type.

    Args:
        model_name (str): The name of the target model.

    Returns:
        Model: The pre-trained model.
    """
    model_builders = {
        "Xception": tf.keras.applications.xception.Xception,
        "ResNet": tf.keras.applications.resnet_v2.ResNet50V2,
    }

    model_builder = model_builders.get(model_name)
    if model_builder:
        model = model_builder(weights="imagenet", classifier_activation="softmax")

    # Deactivate final activation layer for Grad-CAM
    model.layers[-1].activation = None

    fem_model = model

    last_layer_name = get_last_conv_layer(model_name)

    # Create a model for Grad-CAM using the specified last layer
    gradcam_model = Model(
        model.inputs, [model.get_layer(last_layer_name).output, model.output]
    )

    return gradcam_model, fem_model


def resize_array(model_name: str, img_array: np.ndarray) -> np.ndarray:
    """
    Resize an image array to the appropriate size for a given model.

    Args:
        model_name (str): The name of the target model.
        img_array (np.ndarray): The input image array.

    Returns:
        np.ndarray: The resized image.
    """
    size_dict = {"Xception": (299, 299), "ResNet": (224, 224)}
    size = size_dict.get(model_name, (299, 299))
    return resize(img_array, size)


label_index_dictionary = {"Colonial": 0, "Modern": 1, "Prehispanic": 2}


def get_index_from_image(filepath: str) -> int:
    """
    Get the index corresponding to an image based on its label.

    Args:
        filepath (str): The path to the image file.

    Returns:
        int: The index corresponding to the image label.
    """
    filename = os.path.basename(filepath)
    label = filename.split("_")[0]
    index = label_index_dictionary.get(label, -1)  # Default to -1 if label not found

    return index


def plotter(
    image: np.ndarray,
    gt: np.ndarray,
    saliency_map: np.ndarray,
    blended_heatmap: np.ndarray,
    deletion_nval: list,
    deletion_scores: list,
    deletion_auc: float,
    insertion_nval: list,
    insertion_scores: list,
    insertion_auc: float,
    pcc_val: float,
    sim_val: float,
    filename: str,
    explanation: str,
    i: int,
    save=False,
) -> None:
    """
    Plots various visualizations for a given image and explanation.

    Args:
        image (numpy.ndarray): The original image.
        gt (numpy.ndarray): The ground truth.
        saliency_map (numpy.ndarray): The saliency map.
        blended_heatmap (numpy.ndarray): The blended heatmap.
        deletion_nval (list): The number of pixels deleted at each step for the deletion ROC curve.
        deletion_scores (list): The ROC curve scores for deletion.
        deletion_auc (float): The AUC for the deletion ROC curve.
        insertion_nval (list): The number of pixels inserted at each step for the insertion ROC curve.
        insertion_scores (list): The ROC curve scores for insertion.
        insertion_auc (float): The AUC for the insertion ROC curve.
        pcc_val (float): The PCC between the explanation and the ground truth.
        sim_val (float): The similarity between the explanation and the ground truth.
        filename (str): The name of the image file.
        explanation (str): The type of explanation.
        i (int): The index of the image.
        save (bool, optional): Whether to save the plot (default: False).

    Raises:
        ValueError: If the length of `deletion_nval` or `deletion_scores` is not equal to the length of `insertion_nval` or `insertion_scores` respectively.
    """
    _ = plt.figure(figsize=(15, 9))
    gs = GridSpec(3, 4)

    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title("Image")

    ax2 = plt.subplot(gs[0, 1])
    img2 = ax2.imshow(gt, cmap="turbo")
    ax2.axis("off")
    ax2.set_title("GT")
    _ = plt.colorbar(img2, ax=ax2)

    ax3 = plt.subplot(gs[0, 2])
    img3 = ax3.imshow(saliency_map, cmap="turbo")
    ax3.axis("off")
    ax3.set_title("Saliency Map")
    _ = plt.colorbar(img3, ax=ax3)

    ax4 = plt.subplot(gs[0, 3])
    ax4.imshow(blended_heatmap, cmap="turbo")
    ax4.axis("off")
    ax4.set_title("Blended Heatmap")

    ax5 = plt.subplot(gs[1:, 0:2])
    ax5.set_title("Deletion")
    ax5.plot(
        deletion_nval,
        deletion_scores,
        color="royalblue",
        linewidth=2,
        label="ROC Curve",
    )
    ax5.fill_between(deletion_nval, 0, deletion_scores, color="royalblue", alpha=0.2)
    ax5.set_xlabel("Deleted pixels", fontsize=10)
    ax5.set_ylabel("Score", fontsize=10)
    ax5.legend(loc="lower right", fontsize=10)
    ax5.text(
        0.9,
        0.9,
        f"AUC: {deletion_auc:.3f}",
        transform=ax5.transAxes,
        fontsize=10,
        ha="right",
    )
    ax5.grid(True, linestyle="--", alpha=0.6)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.tick_params(axis="both", which="major", labelsize=10)

    ax6 = plt.subplot(gs[1:, 2:])
    ax6.set_title("Insertion")
    ax6.plot(
        insertion_nval,
        insertion_scores,
        color="forestgreen",
        linewidth=2,
        label="ROC Curve",
    )
    ax6.fill_between(
        insertion_nval, 0, insertion_scores, color="forestgreen", alpha=0.2
    )
    ax6.set_xlabel("Inserted pixels", fontsize=10)
    ax6.set_ylabel("Score", fontsize=10)
    ax6.text(
        0.1,
        0.9,
        f"AUC: {insertion_auc:.3f}",
        transform=ax6.transAxes,
        fontsize=10,
        ha="left",
    )
    ax6.legend(loc="lower left", fontsize=10)
    ax6.grid(True, linestyle="--", alpha=0.6)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.tick_params(axis="both", which="major", labelsize=10)

    plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.4)
    plt.suptitle(
        f"{explanation} - {filename}\nPCC = {pcc_val:.4f}, SIM = {sim_val:.4f}"
    )
    if save == True:
        plt.savefig(f"output/{explanation}/{i+1}_{explanation}_{filename}.png")
        plt.close()
    else:
        plt.show()
