import cv2
import os
import argparse
from gsdevice import Camera
from gs3drecon import Reconstruction3D
from gsviz import Visualize3D
import yaml

"""
Show a stream of 3d reconstructed result
"""


def show_3dstream(args):
    """
    Stream GelSight mini 3D data.
    """
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        imgh = config["h"]
        imgw = config["w"]
        ppmm = config["ppmm"]

    # Connect to the camera
    camera = Camera("GelSight Mini", imgh, imgw)
    camera.connect()

    calib_dir = args.calib_dir
    # 3d reconstructioner 
    recon = Reconstruction3D(calib_dir=calib_dir)
    
    # Point cloud visualizer
    viz = Visualize3D(camera.imgh, camera.imgw)

    # Show data until quit
    print(".. hit ESC to exit! .. ")
    while True:
        # Get image
        image = camera.get_image()  # (240, 320, 3)
        # Plot 3D data
        cv2.imshow("frame", image)
        if cv2.waitKey(1) == 27:
            break
        
        # Get surface information
        gradient, heightmap, mask = recon.get_surface_info(image, ppmm)
        # Visualize 3D data
        viz.update(heightmap, gradient[:, :, 0], gradient[:, :, 1])

    cv2.destroyAllWindows()
    viz.close()


if __name__ == "__main__":
    # Use current directory to guess default model path
    lib_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_path = os.path.join(lib_dir, "data/ball_indenters/model/nnmini.pth")
    print(f"model_path: {model_path}")
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Show GelSight Mini 3D data in a stream"
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        default="/home/irving/Desktop/gslib/data",
        help="place where the calibration data is stored",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of configuring gelsight",
        default="/home/irving/Desktop/gslib/configs/gsmini.yaml",
    )
    args = parser.parse_args()
    # Show 3D data stream
    show_3dstream(args)
