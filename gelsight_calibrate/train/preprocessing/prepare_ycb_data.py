import cv2
import os
import os.path as osp
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
import yaml
from gslib.utils import load_csv_as_dict, image2bgrxys
from gslib.gsviz import plot_gradients

"""
Use the simulated ycb data to prepare the universal data npz files.
"""


def prepare_ycb_data(args):
    """
    Prepare universal data npz files using the ycb sim data
    """
    # Load the data_dict
    calib_dir = args.calib_dir
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    data_dict = load_csv_as_dict(catalog_path)
    experiment_reldirs = np.array(data_dict["experiment_reldir"])

    # Split data into train and test and save the split information
    object_names = [reldir.split("/")[0] for reldir in experiment_reldirs]
    unique_object_names = np.unique(object_names)
    perm = np.random.permutation(len(unique_object_names))
    n_train = 4 * len(unique_object_names) // 5
    train_object_names = unique_object_names[perm[:n_train]]
    test_object_names = unique_object_names[perm[n_train:]]
    train_reldirs = [
        reldir
        for reldir in experiment_reldirs
        if reldir.split("/")[0] in train_object_names
    ]
    test_reldirs = [
        reldir
        for reldir in experiment_reldirs
        if reldir.split("/")[0] in test_object_names
    ]
    data_path = osp.join(calib_dir, "train_test_split.json")
    dict_to_save = {"train": train_reldirs, "test": test_reldirs}
    with open(data_path, "w") as f:
        json.dump(dict_to_save, f, indent=4)

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]

    # Create numpy files
    for experiment_reldir in experiment_reldirs:
        # pixel poses and bgrs
        experiment_dir = os.path.join(calib_dir, experiment_reldir)
        image_path = os.path.join(experiment_dir, "gelsight.png")
        image = cv2.imread(image_path)

        # Filter the non-indented pixels
        mask_path = os.path.join(experiment_dir, "contacts.npy")
        mask = np.load(mask_path)

        # Find the gradient angles and prepare the data
        gxy_path = os.path.join(experiment_dir, "gxys.npy")
        gxys = np.load(gxy_path)
        gxyangles = np.arctan(gxys)
        bgrxys = image2bgrxys(image)

        # Visualize and save the prepared data
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        contours = find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color="white")
        plot_gradients(
            fig, ax, np.tan(gxyangles[:, :, 0]), np.tan(gxyangles[:, :, 1]), mask
        )
        plt.axis("off")
        plt.savefig(
            os.path.join(experiment_dir, "labeled_frame.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        # Save data
        save_path = os.path.join(experiment_dir, "data.npz")
        np.savez(save_path, bgrxys=bgrxys, gxyangles=gxyangles, mask=mask)

    # Append with background image data
    bg_path = os.path.join(calib_dir, "background.png")
    bg_image = cv2.imread(bg_path)
    bgrxys = image2bgrxys(bg_image)
    gxyangles = np.zeros((bg_image.shape[0], bg_image.shape[1], 2))
    mask = np.ones((bg_image.shape[0], bg_image.shape[1]), dtype=np.bool_)

    # Save data
    save_path = os.path.join(calib_dir, "background_data.npz")
    np.savez(save_path, bgrxys=bgrxys, gxyangles=gxyangles, mask=mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use the labeled GelSight mini ball data to prepare the universal data npz files."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="path to save calibration data",
        default="/home/joehuang/research/gelsightfusion/data/calibration/gelsight2/ycb_data",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of configuring gelsight",
        default="/home/joehuang/research/gelsightfusion/tactile_odometry/configs/gsmini.yaml",
    )
    args = parser.parse_args()
    # Prepare universal data npz files using the labeled ycb data
    prepare_ycb_data(args)
