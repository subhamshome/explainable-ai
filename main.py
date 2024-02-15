import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize

import config
from fem import *
from rise import *
from lime_ import *
from utils import *
from gradcam import *
from metrics import *
from representation import *


def main():
    """
    Main function to perform image explanation using different techniques and evaluate the results.
    """

    # Load the pretrained model
    model = tf.keras.models.load_model("resnet50_tl_mex.h5", compile=False)

    n_masks = config.N_MASKS
    probability = config.PROBABILITY
    image_size = config.IMAGE_SIZE
    n_classes = config.N_CLASSES
    mask_dim = config.MASK_DIM
    top_labels = config.TOP_LABELS
    hide_color = config.HIDE_COLOR
    num_lime_features = config.NUM_LIME_FEATURES
    num_samples = config.NUM_SAMPLES

    positive_only = config.POSITIVE_ONLY
    negative_only = config.NEGATIVE_ONLY
    num_superpixels = config.NUM_SUPERPIXELS
    hide_rest = config.HIDE_REST
    class_name = "monastery"

    parser = argparse.ArgumentParser(description="Explainable methods and metrics")

    parser.add_argument(
        "--input_folder", type=str, default="input", help="Original images"
    )
    parser.add_argument(
        "--gt_folder", type=str, default="gt", help="Ground truth images"
    )
    parser.add_argument(
        "--explanation",
        default="gradcam",
        type=str,
        help="Choose between gradcam and fem",
    )
    parser.add_argument("--colormap", default="turbo", help="Colormap")
    parser.add_argument("--alpha", default=0.5, type=float, help="Blending factor")
    parser.add_argument("--save", default=False, type=bool, help="If save or not")

    args = parser.parse_args()

    img_folder = args.input_folder
    gt_folder = args.gt_folder
    model_name = args.model_name
    colormap = args.colormap
    alpha = args.alpha
    explanation = args.explanation
    image_arrays = []
    saliency_maps = []
    pcc_vals = []
    sim_vals = []
    deletion_vals = []
    insertion_vals = []
    save = args.save

    for i, filename in enumerate(os.listdir(img_folder)):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            print(f"Image {i+1} started.")
            image_path = os.path.join(img_folder, filename)
            image = Image.open(image_path)
            image_size = image.size
            image_arrays.append(filename)

            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = resize_array(model_name, image_array)
            gt_filename = filename.replace("_N_", "_GFDM_N_")
            gt_path = os.path.join(gt_folder, gt_filename)
            gt = np.array(Image.open(gt_path))
            gt = (gt - gt.min()) / (gt.max() - gt.min())

            if explanation == "fem":
                saliency_map = compute_fem(image_array, model_name)
                predict(model_name, image_array)
            elif explanation == "gradcam":
                grads, last_layer = compute_gradients(
                    model_name, model, class_name, image_array, image_path
                )
                pooled_grads = pool_gradients(grads)
                weighted_maps = calculate_weighted_activation_maps(
                    last_layer, pooled_grads
                )
                weighted_maps = relu(weighted_maps)
                saliency_map = apply_dimension_average_pooling(weighted_maps)
            elif explanation == "rise":
                masks = generate_masks(n_masks, image_array, probability, mask_dim)
                perturbed_images = calculate_perturbed_images(image_array, masks)
                prediction_scores = compute_predictions(
                    perturbed_images, n_classes, class_name, image_path
                )
                saliency_map = calculate_saliency_map(prediction_scores, masks)
            elif explanation == "lime":
                img_array = preprocess_img_array(model_name, image_array)
                saliency_map = explain_with_lime(
                    img_array,
                    top_labels,
                    hide_color,
                    num_lime_features,
                    num_samples,
                    positive_only,
                    negative_only,
                    num_superpixels,
                    hide_rest,
                    model_name,
                )

            # Normalize and resize saliency map
            saliency_map = saliency_map / np.max(saliency_map)

            saliency_map_for_del_ins = resize(
                saliency_map,
                config.IMAGE_SIZE,
                order=3,
                mode="wrap",
                anti_aliasing=False,
            )

            saliency_map = resize(
                saliency_map,
                (image_size[1], image_size[0]),
                order=3,
                mode="wrap",
                anti_aliasing=False,
            )

            # Generate overlaid heatmap
            blended_heatmap = represent_heatmap_overlaid(
                saliency_map, image, alpha, colormap
            )

            # Perform deletion and insertion algorithms
            deletion_auc, deletion_scores, deletion_nval = deletion(
                image_array, image_path, saliency_map_for_del_ins, 500
            )
            deletion_vals.append(deletion_auc)

            insertion_auc, insertion_scores, insertion_nval = insertion(
                image_array, image_path, saliency_map_for_del_ins, 500
            )
            insertion_vals.append(insertion_auc)

            # Calculate similarity metrics
            sim_val = calculate_sim(gt, saliency_map)
            pcc_val = calculate_pcc(gt, saliency_map)
            pcc_vals.append(pcc_val)
            sim_vals.append(sim_val)

            # Plot and save results
            plotter(
                image,
                gt,
                saliency_map,
                blended_heatmap,
                deletion_nval,
                deletion_scores,
                deletion_auc,
                insertion_nval,
                insertion_scores,
                insertion_auc,
                pcc_val,
                sim_val,
                filename,
                explanation,
                i,
                save=save,
            )

            saliency_maps.append(saliency_map)
            print(f"Image {i+1} done.")

    # Calculate and print mean and standard deviation of evaluation metrics
    mean_pcc = np.mean(np.array(pcc_vals))
    std_pcc = np.std(np.array(pcc_vals))
    mean_ssim = np.mean(np.array(sim_vals))
    std_ssim = np.std(np.array(sim_vals))
    mean_del = np.mean(np.array(deletion_vals))
    std_del = np.std(np.array(deletion_vals))
    mean_ins = np.mean(np.array(insertion_vals))
    std_ins = np.std(np.array(insertion_vals))
    print(f"PCC: {mean_pcc} +- {std_pcc}")
    print(f"SSIM: {mean_ssim} +- {std_ssim}")
    print(f"Deletion: {mean_del} +- {std_del}")
    print(f"Insertion: {mean_ins} +- {std_ins}")

    # Specify the file path
    file_path = f"output/{explanation}/{explanation}_output_vals.txt"

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write the values to the file
        file.write(f"PCC: {mean_pcc} +- {std_pcc}\n")
        file.write(f"SSIM: {mean_ssim} +- {std_ssim}\n")
        file.write(f"Deletion: {mean_del} +- {std_del}\n")
        file.write(f"Insertion: {mean_ins} +- {std_ins}\n")

    print(f"Results for {explanation} saved to {file_path}")


if __name__ == "__main__":
    main()
