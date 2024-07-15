import argparse
import cv2
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import shutil
from os import path as osp
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from tactile_odometry.utils import rodrigues
from tactile_odometry.data_collection.simulator import Simulator
from Taxim.OpticalSimulation.simOptical import TaximSimulator

"""
Collect Simulation data using ycb models.
It will save gxy, height, contact mask, and Taxim simulated data for each simulated frame.
"""
obj_names = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "017_orange",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "029_plate",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "048_hammer",
    "055_baseball",
    "056_tennis_ball",
    "072-a_toy_airplane",
    "072-b_toy_airplane",
    "077_rubiks_cube",
]


def collect_ycb(args):
    """
    Simulate the gxy, height, contact mask, and Taxim simulated data for each simulated frame.
    """
    # Load the arguments
    ycb_dir = args.ycb_dir
    taxim_calib_dir = args.taxim_calib_dir
    calib_dir = args.calib_dir
    config_path = args.config_path
    if not os.path.isdir(calib_dir):
        os.makedirs(calib_dir)
    # Read the configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Construct the Taxim Simulator
    taxsim = TaximSimulator(taxim_calib_dir, config)

    # Copy the background image from the taxim_calib_dir to the calib_dir
    data_path = os.path.join(taxim_calib_dir, "frame_0.jpg")
    save_path = os.path.join(calib_dir, "background.png")
    shutil.copy(data_path, save_path)

    # Create the data saving catalog
    if not osp.exists(calib_dir):
        os.makedirs(calib_dir)
    catalog_path = osp.join(calib_dir, "catalog.csv")
    with open(catalog_path, "w") as f:
        f.write("experiment_reldir\n")

    # Load initial pressing poses
    for obj_name in obj_names:
        # The object model and the simulator
        ply_path = osp.join(ycb_dir, "models", obj_name, "google_512k", "nontextured.ply")
        sim = Simulator(ply_path, config)
        # Load pressing poses
        data_path = osp.join(ycb_dir, "models", obj_name, "textured_60sampled_python.mat")
        data_dict = loadmat(data_path)
        pressing_points = data_dict["samplePoints"]
        pressing_normals = data_dict["sampleNormals"]
        for pressing_idx in range(len(pressing_points)):
            pressing_point = pressing_points[pressing_idx]
            pressing_normal = pressing_normals[pressing_idx]
            press_depth = np.random.uniform(0.0007, 0.0013)
            # Create the point to world transformation
            z_axis = np.array([0, 0, 1])
            Tp2w = np.zeros((4, 4))
            Tp2w[3, 3] = 1
            Tp2w[0:3, 3] = pressing_point
            v = np.cross(z_axis, -pressing_normal)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, -pressing_normal)
            Rot = rodrigues(v, s, c)
            Tp2w[0:3, 0:3] = Rot
            # Pressing down
            Tp2p = np.identity(4)
            Tp2p[2, 3] = press_depth
            Tp2w = np.dot(Tp2w, Tp2p)

            # Create the directory for this experiment
            experiment_dir = osp.join(calib_dir, obj_name, str(pressing_idx))
            experiment_reldir = osp.join(obj_name, str(pressing_idx))
            if not osp.isdir(experiment_dir):
                os.makedirs(experiment_dir)

            # Simulate true readings
            H, C, I = sim.simulate(Tp2w, viz_mode="None")
            np.save(osp.join(experiment_dir, "heights.npy"), H)
            np.save(osp.join(experiment_dir, "contacts.npy"), C)
            np.save(osp.join(experiment_dir, "gxys.npy"), I)
            # Save height image
            plt.imshow(H, cmap="gray")
            plt.colorbar()
            plt.savefig(osp.join(experiment_dir, "height.png"))
            plt.close()

            # generate Taxim simulated image
            F = taxsim.simulate(H, C)
            cv2.imwrite(osp.join(experiment_dir, "gelsight.png"), F)

            # Save to catalog
            with open(catalog_path, "a") as f:
                f.write(str(experiment_reldir))
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect 3D data using simulation.")
    parser.add_argument(
        "-t",
        "--taxim_calib_dir",
        type=str,
        help="path to taxim calibration data",
        default="/home/joehuang/research/gelsightfusion/data/taxim/gelsight2/03_20_2024/ball5.0",
    )
    parser.add_argument(
        "-y",
        "--ycb_dir",
        type=str,
        help="path to ycb models",
        default="/home/joehuang/research/gelsightfusion/data/ycb",
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
        help="path of configuring gelsight simulator",
        default="/home/joehuang/research/gelsightfusion/tactile_odometry/configs/gsmini.yaml",
    )
    args = parser.parse_args()
    collect_ycb(args)
