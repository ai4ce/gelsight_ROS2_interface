import cv2
import numpy as np
import os
import re
import asyncio
import subprocess

if os.name == 'nt':
    import winrt.windows.devices.enumeration as windows_devices

VIDEO_DEVICES = 4 # video device is labelled as 4 in windows

def get_camera_id(camera_name):
    """Find the camera ID that has the corresponding camera name."""
    cam_num = None
    if os.name == 'nt':
        cam_num = find_cameras_windows(camera_name)
    else:
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            if camera_name in name:
                cam_num = int(re.search("\d+$", file).group(0))
                found = "FOUND!"
            else:
                found = "      "
            print("{} {} -> {}".format(found, file, name))

    return cam_num

if os.name == 'nt':
    def find_cameras_windows(camera_name):
        allcams = get_camera_information(get_camera_indexes())# list of camera device
        
        if len(allcams) != 0:
            for cam in allcams:
                if camera_name in cam['camera_name']:
                    print("IGNORE PREVIOUS WARNING. It was just searching through all USB ports for the sensor.")
                    return cam['camera_index']
        
        print("Device is not in this list")
        print(allcams)
        import sys
        sys.exit()

    
    def get_camera_indexes():
        camera_indexes = []
    
        # the total number of possible USB camera is less than the total number of USBHub
        max_numbers_of_cameras_to_check = int(subprocess.getoutput("PowerShell -Command \"& {@(gwmi Win32_USBHub).count}\""))
        for index in range(max_numbers_of_cameras_to_check):
            capture = cv2.VideoCapture(index)
            if capture.read()[0]: # check if there is a camera connected to the index
                camera_indexes.append(index)
                capture.release()
        return camera_indexes
    
    def get_camera_information(camera_indexes: list) -> list:
        cameras_info = []

        cameras_info_windows = asyncio.run(get_camera_information_for_windows())

        for camera_index in camera_indexes:
            try:
                camera_name = cameras_info_windows.get_at(camera_index).name.replace('\n', '')
                cameras_info.append({'camera_index': camera_index, 'camera_name': camera_name})
            except:
                continue
        return cameras_info
    
    async def get_camera_information_for_windows():
        return await windows_devices.DeviceInformation.find_all_async(VIDEO_DEVICES)
    
def resize_crop(img, imgw, imgh):
    """Resize and crop the image to the desired size."""
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
        np.floor(img.shape[1] * (1 / 7))
    )
    cropped_imgh = img.shape[0] - 2 * border_size_x
    cropped_imgw = img.shape[1] - 2 * border_size_y
    # Extra cropping to maintain aspect ratio
    extra_border_h = 0
    extra_border_w = 0
    if cropped_imgh * imgw / imgh > cropped_imgw + 1e-8:
        extra_border_h = int(cropped_imgh - cropped_imgw * imgh / imgw)
    elif cropped_imgh * imgw / imgh < cropped_imgw - 1e-8:
        extra_border_w = int(cropped_imgw - cropped_imgh * imgw / imgh)
    # keep the ratio the same as the original image size
    img = img[
        border_size_x + extra_border_h : img.shape[0] - border_size_x,
        border_size_y + extra_border_w : img.shape[1] - border_size_y,
    ]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img



class Camera:
    """The GelSight Camera Class."""
    def __init__(self, dev_type, imgh, imgw):
        # variable to store data
        self.data = None
        self.name = dev_type
        self.dev_id = get_camera_id(dev_type)
        self.imgh = imgh
        self.imgw = imgw
        self.cam = None
        self.while_condition = 1

    def connect(self):
        """Connect to the camera using cv2 streamer."""
        self.cam = cv2.VideoCapture(self.dev_id)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cam is None or not self.cam.isOpened():
            print("Warning: unable to open video source: ", self.dev_id)

        return self.cam

    def get_image(self, flush=False):
        """Get the image from the camera."""
        if flush:
            # flush out fist few frames to remove black frames
            for i in range(10):
                ret, f0 = self.cam.read()
        ret, f0 = self.cam.read()
        if ret:
            f0 = resize_crop(f0, self.imgw, self.imgh)
        else:
            print("ERROR! reading image from camera!")
        self.data = f0
        return self.data