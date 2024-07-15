import cv2
import os
import argparse
import numpy as np
import yaml
from gslib.gsdevice import Camera
from gslib.utils import load_csv_as_dict

"""
Collect gelsight data using random objects in real world
This is just for testing and will not obtain ground truth results
"""


def collect_random_objects(args):
    """
    Collect GelSight data using random objects in real world
    """
    # Create the data saving directories
    calib_dir = args.calib_dir
    if not os.path.isdir(calib_dir):
        os.makedirs(calib_dir)
    object_subdir = None

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
            f.write("experiment_reldir,object_name\n")

    # Connect to the device
    device = Camera(device_name, h, w)
    device.connect()

    # Collect data until quit
    data_count = 0
    print("Press key to change new object, collect data, collect background, or quit (n/w/b/q)")
    while True:
        # Get image
        image = device.get_image()

        # Display the image and decide record or quit
        cv2.imshow("frame", image)
        key = cv2.waitKey(100)
        if key == ord("n"):
            object_name = input("Enter the object name: ")
            object_subdir = object_name
            data_count = 0
        elif key == ord("w"):
            if object_subdir is None:
                print("Please enter the object name first")
                continue
            # Save the image
            experiment_reldir = os.path.join(object_subdir, str(data_count))
            experiment_dir = os.path.join(calib_dir, experiment_reldir)
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)
            save_path = os.path.join(experiment_dir, "gelsight.png")
            cv2.imwrite(save_path, image)
            print("Save data to new path: %s" % save_path)

            # Save to catalog
            with open(catalog_path, "a") as f:
                f.write(experiment_reldir + "," + object_name)
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
        description="Collect GelSight Mini data using random objects to train network."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="path to save calibration data",
        default="/home/joehuang/research/gelsightfusion/data/calibration/gelsight2/random_objects_data",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of configuring gelsight",
        default="/home/joehuang/research/gelsightfusion/tactile_odometry/configs/gsmini.yaml",
    )
    args = parser.parse_args()
    # Collect data using random real objects
    collect_random_objects(args)
