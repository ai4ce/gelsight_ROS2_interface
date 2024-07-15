"""
this is the file to pre-process the raw data to the data pack
the data pack including: images, touch center, touch radius

Keyboard control:
left/right/up/down: control the circle's location
m/p: decrease/increase the circle's radius
f/c: decrease/increase the circle's moving step
"""

import gc
from os import path as osp
import cv2
import numpy as np
import argparse
import yaml

import nanogui as ng
from nanogui import Texture
from nanogui import glfw
from gelsight_calibrate.utils import load_csv_as_dict


class Circle:
    """the circle drawed on the tactile image to get the contact size"""

    color_circle = (128, 0, 0)
    opacity = 0.2

    def __init__(self, x, y, radius=25, increments=2):
        self.center = [x, y]
        self.radius = radius
        self.increments = increments


class CalibrateApp(ng.Screen):
    fnames = list()
    read_all = False  # flag to indicate if all images have been read
    load_img = True
    change = False

    def __init__(self, calib_data, imgw, imgh):
        super(CalibrateApp, self).__init__((1024, 768), "Gelsight Calibration App")
        self.imgw = imgw
        self.imgh = imgh
        # Load background 
        self.bg_img = cv2.imread(osp.join(calib_data, "background.png"))
        # Initialize the circle
        self.circle = Circle(self.imgw/2, self.imgh/2, radius=40)

        window = ng.Window(self, "IO Window")
        window.set_position((15, 15))
        window.set_layout(ng.GroupLayout())

        ng.Label(window, "Folder dialog", "sans-bold")
        tools = ng.Widget(window)
        tools.set_layout(
            ng.BoxLayout(ng.Orientation.Horizontal, ng.Alignment.Middle, 0, 6)
        )

        # Initialize the file directory and list of filenames
        b = ng.Button(tools, "Open")

        def open_cb():
            self.parent_dir = calib_data
            # Read the catalog and create a list of all filenames
            catalog_dict = load_csv_as_dict(osp.join(self.parent_dir, "catalog.csv"))
            self.fnames = [
                osp.join(self.parent_dir, fname)
                for fname in catalog_dict["experiment_reldir"]
            ]
            self.circle_radius = [
                float(radius) for radius in catalog_dict["diameter(mm)"]
            ]
            print(
                f"Selected directory = {self.parent_dir}, total {len(self.fnames)} images"
            )
            self.img_idx = 0

        b.set_callback(open_cb)

        # Initialize the image window
        self.img_window = ng.Window(self, "Current image")
        self.img_window.set_position((200, 15))
        self.img_window.set_layout(ng.GroupLayout())

        # Initialize the calibrate button
        b = ng.Button(self.img_window, "Calibrate")

        def calibrate_cb():
            frame = self.orig_img
            center = self.circle.center
            radius = self.circle.radius
            print(f"Frame {self.img_idx}: center = {center}, radius = is {radius}")
            # save the data for each individual frame instead of creating a long list and save it at the end
            # save the radius and center to npz file
            save_dir = osp.join(self.fnames[self.img_idx], "label.npz")
            np.savez(save_dir, center=center, radius=radius)
            # save the labeled image
            labeled_img = self.overlay_circle(frame, self.circle)
            labeled_img_path = osp.join(self.fnames[self.img_idx], "labeled.png")
            cv2.imwrite(labeled_img_path, labeled_img)

            # Update img index
            self.load_img = True
            self.update_img_idx()

        b.set_callback(calibrate_cb)

        ###########
        self.img_view = ng.ImageView(self.img_window)
        self.img_tex = ng.Texture(
            pixel_format=Texture.PixelFormat.RGB,
            component_format=Texture.ComponentFormat.UInt8,
            size=[imgw, imgh],
            min_interpolation_mode=Texture.InterpolationMode.Trilinear,
            mag_interpolation_mode=Texture.InterpolationMode.Nearest,
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget,
        )
        self.perform_layout()

    def update_img_idx(self):
        self.img_idx += 1
        if self.img_idx == len(self.fnames) - 1:
            self.read_all = True

    def overlay_circle(self, orig_img, circle):
        center = circle.center
        radius = circle.radius
        color_circle = circle.color_circle
        opacity = circle.opacity

        overlay = orig_img.copy()
        center_tuple = (int(center[0]), int(center[1]))
        cv2.circle(overlay, center_tuple, radius, color_circle, -1)
        cv2.addWeighted(overlay, opacity, orig_img, 1 - opacity, 0, overlay)
        return overlay

    def draw(self, ctx):
        self.img_window.set_size((2000, 2600))
        self.img_view.set_size((self.imgw, self.imgh))

        # load a new image
        if self.load_img and len(self.fnames) > 0 and not self.read_all:
            print("Loading %s" % self.fnames[self.img_idx])

            # Load img
            self.orig_img = cv2.imread(
                osp.join(self.fnames[self.img_idx], "gelsight.png")
            )
            # Initialize the circle pose
            diff_image = self.orig_img.astype(np.float32) - self.bg_img.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > 15
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
            )
            color_mask = cv2.erode(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            radius = max(self.circle.radius - 13, 5)
            self.circle = Circle(cx, cy, radius=radius)


        # Add circle and add img to viewer
        if (self.load_img and len(self.fnames) > 0) or self.change:
            self.load_img = False
            self.change = False
            # Add circle
            diff_img = (self.orig_img.astype(np.float32) - self.bg_img.astype(np.float32)) * 3
            diff_img = np.clip(diff_img, -127, 128) + np.ones_like(diff_img) * 127
            display_img = cv2.cvtColor(diff_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            img = self.overlay_circle(display_img, self.circle)

            if self.img_tex.channels() > 3:
                height, width = img.shape[:2]
                alpha = 255 * np.ones((height, width, 1), dtype=img.dtype)
                img = np.concatenate((img, alpha), axis=2)

            # Add to img view
            self.img_tex.upload(img)
            self.img_view.set_image(self.img_tex)

        super(CalibrateApp, self).draw(ctx)

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(CalibrateApp, self).keyboard_event(key, scancode, action, modifiers):
            return True
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.set_visible(False)
            return True
        elif key == glfw.KEY_C:
            self.circle.increments *= 2
        elif key == glfw.KEY_F:
            self.circle.increments /= 2
        else:
            self.change = True
            if key == glfw.KEY_LEFT:
                self.circle.center[0] -= self.circle.increments
            elif key == glfw.KEY_RIGHT:
                self.circle.center[0] += self.circle.increments
            elif key == glfw.KEY_UP:
                self.circle.center[1] -= self.circle.increments
            elif key == glfw.KEY_DOWN:
                self.circle.center[1] += self.circle.increments
            elif key == glfw.KEY_M:
                self.circle.radius -= 1
            elif key == glfw.KEY_P:
                self.circle.radius += 1

        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label the ball indenter data using Nanogui."
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
    args = parser.parse_args()
    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        imgh = config["h"]
        imgw = config["w"]

    ng.init()
    app = CalibrateApp(args.calib_dir, imgw, imgh)
    app.draw_all()
    app.set_visible(True)
    ng.mainloop(refresh=1 / 60.0 * 1000)
    del app
    gc.collect()
    ng.shutdown()
