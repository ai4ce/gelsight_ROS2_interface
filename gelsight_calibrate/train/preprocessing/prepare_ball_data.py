import cv2
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import os.path as osp
from skimage.measure import find_contours
import yaml
from gelsight_calibrate.utils import load_csv_as_dict, image2bgrxys
from gelsight_calibrate.gsviz import plot_gradients

"""
Use the labeled ball data to prepare the universal data npz files.
"""


def prepare_ball_data(args):
    """
    Prepare universal data npz files using the labeled ball data
    """
    # Load the data_dict
    calib_dir = args.calib_dir
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    data_dict = load_csv_as_dict(catalog_path)
    diameters = np.array([float(diameter) for diameter in data_dict["diameter(mm)"]])
    experiment_reldirs = np.array(data_dict["experiment_reldir"])

    # Split data into train and test and save the split information
    perm = np.random.permutation(len(experiment_reldirs))
    n_train = 4 * len(experiment_reldirs) // 5
    data_path = osp.join(calib_dir, "train_test_split.json")
    dict_to_save = {
        "train": experiment_reldirs[perm[:n_train]].tolist(),
        "test": experiment_reldirs[perm[n_train:]].tolist(),
    }
    with open(data_path, "w") as f:
        json.dump(dict_to_save, f, indent=4)

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]

    # Get all pixel poses, bgr, gradient angles
    for experiment_reldir, diameter in zip(experiment_reldirs, diameters):
        # pixel poses and bgrs
        experiment_dir = os.path.join(calib_dir, experiment_reldir)
        image_path = os.path.join(experiment_dir, "gelsight.png")
        image = cv2.imread(image_path)

        # Filter the non-indented pixels
        label_path = os.path.join(experiment_dir, "label.npz")
        label_data = np.load(label_path)
        center = label_data["center"]
        radius = label_data["radius"] - args.radius_reduction
        xys = np.dstack(
            np.meshgrid(
                np.arange(image.shape[1]), np.arange(image.shape[0]), indexing="xy"
            )
        )
        dists = np.linalg.norm(xys - center, axis=2)
        mask = dists < radius

        # Find the gradient angles and prepare the data
        ball_radius = diameter / ppmm / 2.0
        if ball_radius < radius:
            print("Press too deep, deeper than the ball radius")
            continue
        dxys = xys - center
        dists[np.logical_not(mask)] = 0.0
        dzs = np.sqrt(ball_radius**2 - np.square(dists))
        gxangles = np.arctan2(dxys[:, :, 0], dzs)
        gyangles = np.arctan2(dxys[:, :, 1], dzs)
        gxyangles = np.stack([gxangles, gyangles], axis=-1)
        gxyangles[np.logical_not(mask)] = np.array([0.0, 0.0])
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
    mask = np.ones(
        (bg_image.shape[0], bg_image.shape[1]), dtype=np.bool_
    )

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
        default="/home/zf540/tactile_recon_ws/src/gelsight_ROS2_interface/gelsight_calibrate/data",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of configuring gelsight",
        default="/home/zf540/tactile_recon_ws/src/gelsight_ROS2_interface/gelsight_calibrate/configs/gsmini.yaml",
    )
    parser.add_argument(
        "-r",
        "--radius_reduction",
        type=float,
        help="reduce the radius of the label. When not considering shadows, this helps guarantee all labeled pixels are indented. ",
        default=0,
    )
    args = parser.parse_args()
    # Prepare universal data npz files using the labeled ball data
    prepare_ball_data(args)
