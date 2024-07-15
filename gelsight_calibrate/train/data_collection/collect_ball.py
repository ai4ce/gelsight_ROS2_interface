import cv2
import os
import argparse
import numpy as np
import yaml
from gelsight_calibrate.gsdevice import Camera
from gelsight_calibrate.utils import load_csv_as_dict

"""
Collect gelsight data using ball indenters
"""


def collect_ball(args):
    """
    Collect GelSight data when indenting with ball indenter.
    """
    # Create the data saving directories
    calib_dir = args.calib_dir
    ball_diameter = args.ball_diameter
    indenter_subdir = "%.3fmm" % (ball_diameter)
    indenter_dir = os.path.join(calib_dir, indenter_subdir)
    if not os.path.isdir(indenter_dir):
        os.makedirs(indenter_dir)

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        h = config["h"]
        w = config["w"]

    # Create the data saving catalog
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    if not os.path.isfile(catalog_path):
        with open(catalog_path, "w") as f:
            f.write("experiment_reldir,diameter(mm)\n")

    # Find last data_count collected with this diameter
    data_dict = load_csv_as_dict(catalog_path)
    diameters = np.array([float(diameter) for diameter in data_dict["diameter(mm)"]])
    data_idxs = np.where(np.abs(diameters - ball_diameter) < 1e-3)[0]
    data_counts = np.array(
        [int(os.path.basename(reldir)) for reldir in data_dict["experiment_reldir"]]
    )
    if len(data_idxs) == 0:
        data_count = 0
    else:
        data_count = max(data_counts[data_idxs]) + 1

    # Connect to the device
    device = Camera(device_name, h, w)
    device.connect()

    # Collect data until quit
    print("Press key to collect data, collect background, or quit (w/b/q)")
    while True:
        # Get image
        image = device.get_image()

        # Display the image and decide record or quit
        cv2.imshow("frame", image)
        key = cv2.waitKey(100)
        if key == ord("w"):
            # Save the image
            experiment_reldir = os.path.join(indenter_subdir, str(data_count))
            experiment_dir = os.path.join(calib_dir, experiment_reldir)
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)
            save_path = os.path.join(experiment_dir, "gelsight.png")
            cv2.imwrite(save_path, image)
            print("Save data to new path: %s" % save_path)

            # Save to catalog
            with open(catalog_path, "a") as f:
                f.write(experiment_reldir + "," + str(ball_diameter))
                f.write("\n")
            data_count += 1
        elif key == ord("b"):
            print("Collecting 10 background images, please wait ...")
            images = []
            for _ in range(10):
                image = device.get_image()
                images.append(image)
                cv2.imshow("frame", image)
                cv2.waitKey(1)
            image = np.mean(images, axis=0).astype(np.uint8)
            # Save the background image
            save_path = os.path.join(calib_dir, "background.png")
            cv2.imwrite(save_path, image)
            print("Save background image to %s" % save_path)
        elif key == ord("q"):
            # Quit
            break
        elif key == -1:
            # No key pressed
            continue
        else:
            print("Unrecognized key %s" % key)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect GelSight Mini data using ball indenter to train network."
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
        "-d", 
        "--ball_diameter", 
        type=float, 
        help="diameter of the indenter in mm",
        default=6.35
    )
    args = parser.parse_args()
    # Collect data using ball indenters
    collect_ball(args)
