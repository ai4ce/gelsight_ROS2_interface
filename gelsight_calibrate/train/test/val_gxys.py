import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from os import path as osp
from gslib.gs3drecon import Reconstruction3D
from tactile_odometry.utils import plot_colored_gradients

"""
Taken the validation data and show predicted versus real gxy images.
It will save the image comparisons in each frame
"""


def val_gxys(args):
    """
    Validate the GelSight images and visualize the gxy images.
    """
    # Load the arguments
    calib_dir = args.calib_dir
    gxy_mode = args.gxy_mode
    device = args.device
    bg_image = cv2.imread(osp.join(calib_dir, "background.png"))
    model_dir = osp.join(calib_dir, gxy_mode + "_gxy_model")
    split_path = osp.join(calib_dir, "train_test_split.json")
    with open(split_path, "r") as f:
        experiment_reldirs = json.load(f)["test"]
    # Make result directory
    result_dir = osp.join(model_dir, "val_results")
    if not osp.exists(result_dir):
        os.makedirs(result_dir)
    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]
    # The reconstruct object
    recon3d = Reconstruction3D(calib_dir, gxy_mode=gxy_mode, device=device)
    recon3d.load_bg(bg_image)

    # Reconstruct background frame
    I, _, _ = recon3d.get_surface_info(bg_image, ppmm)
    gt_I = np.zeros_like(I)
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    plot_colored_gradients(fig, ax, I[:, :, 0], I[:, :, 1])
    plt.axis("off")
    plt.title("Predicted Gradients")
    ax = fig.add_subplot(122)
    plot_colored_gradients(fig, ax, gt_I[:, :, 0], gt_I[:, :, 1])
    plt.axis("off")
    plt.title("True Gradients")
    plt.savefig(osp.join(result_dir, "background.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

    # Loop through each frames
    for experiment_reldir in experiment_reldirs:
        experiment_dir = osp.join(calib_dir, experiment_reldir)
        image = cv2.imread(osp.join(experiment_dir, "gelsight.png"))
        # Get surface information
        I, _, _ = recon3d.get_surface_info(image, ppmm)
        # Get true gxys
        data = np.load(osp.join(experiment_dir, "data.npz"))
        gt_I = np.tan(data["gxyangles"])

        # Visualize the processed image
        fig = plt.figure(figsize=(24, 6))
        ax = fig.add_subplot(131)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Tactile Image")
        ax = fig.add_subplot(132)
        plot_colored_gradients(fig, ax, I[:, :, 0], I[:, :, 1])
        plt.axis("off")
        plt.title("Estimated Gradients")
        ax = fig.add_subplot(133)
        plot_colored_gradients(fig, ax, gt_I[:, :, 0], gt_I[:, :, 1])
        plt.axis("off")
        plt.title("Ground Truth Gradients")
        plt.savefig(
            osp.join(result_dir, experiment_reldir.replace("/", "_") + ".png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Taken the validation data and show predicted versus real gxy images."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        default="/home/rpl/joehuang/data/calibration/gelsigh2/ball_data",
        help="place where the calibration data is stored",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="the device to train NN",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of configuring gelsight",
        default="/home/joehuang/research/gelsightfusion/tactile_odometry/configs/gsmini.yaml",
    )
    parser.add_argument(
        "-gm",
        "--gxy_mode",
        type=str,
        choices=["mlp", "mlp-nobg", "unet", "unet-nobg", "runet"],
        default="unet",
        help="The method to train the network",
    )
    args = parser.parse_args()
    # Validate the gxys
    val_gxys(args)
