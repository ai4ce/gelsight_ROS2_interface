import cv2
from cv_bridge import CvBridge

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud2, PointField
from gelsight_interface_msg.srv import GSCapture

import sensor_msgs_py.point_cloud2 as pc_utils



from .gs3drecon import Reconstruction3D


FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

class GelsightPointcloudServer(Node):
    def __init__(self):
        super().__init__('gelsight_pointcloud_client') # type: ignore
        
        ############################ Launch Parameters ########################################
        # it's complicated why we need to load the config path instead of the content of config. See launch file for explanation
        self.declare_parameter(name = 'config_path', value = '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value

        # path to the model weights and background image
        self.declare_parameter(name = 'resource_path', value = '')
        resource_path = self.get_parameter('resource_path').get_parameter_value().string_value

        ############################ Miscanellous Setup ########################################
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            self.imgh = config["h"]
            self.imgw = config["w"]
            self.mmpp = config["mmpp"]
        self.multithread_group = ReentrantCallbackGroup()
        self.cvbridge = CvBridge() # for converting ROS images to OpenCV images
        

        # 3d reconstructioner 
        self.recon = Reconstruction3D(resource_dir=resource_path)
    
        # Point cloud visualizer
        # self.viz = Visualize3D(self.imgh, self.imgw)

        # initialize the data we will save
        self._pointcloud_init()
        self._multiarray_init()
        self.image = np.zeros([self.imgh, self.imgw, 3])
        self.normal = np.zeros([self.imgh, self.imgw, 2])

        ############################ Publisher Setup ###########################################
        self.patch_publisher = self.create_publisher(
            msg_type=PointCloud2, 
            topic='/gelsight_capture/patch', 
            qos_profile=10)
        
        self.mask_publisher = self.create_publisher(
            msg_type=PointCloud2,
            topic='/gelsight_capture/mask',
            qos_profile=10)
        
        ############################ Subscriber Setup ###########################################
        self.image_subscriber = self.create_subscription(
            msg_type=Image, 
            topic='/gelsight_capture/image', 
            callback=self.image_sub_callback,
            qos_profile=10,
            callback_group=self.multithread_group)
        
        ############################ Service Setup ###########################################
        self.pointcloud_service = self.create_service(
            srv_type=GSCapture, 
            srv_name='/gelsight_capture/capture', 
            callback=self.get_pointcloud_callback,
            callback_group=self.multithread_group)
        # pointcloud_timer_period = 0.05  # in seconds. equal to 20 Hz
        # self.image_timer = self.create_timer(
        #     timer_period_sec=pointcloud_timer_period, 
        #     callback=self.img_timer_callback,
        #     callback_group=multithread_group)
    
        
    def image_sub_callback(self, msg: Image) -> None:
        original_image = self.cvbridge.imgmsg_to_cv2(img_msg=msg, desired_encoding='passthrough')
        gradient, heightmap, mask = self.recon.get_surface_info(original_image, self.mmpp)
        self.image = msg
        self.normal = gradient
        self._update_pointcloud(heightmap, 'patch')
        self._update_pointcloud(mask, 'mask')
        self.patch_publisher.publish(self.patch)
        self.mask_publisher.publish(self.mask)
    
    def get_pointcloud_callback(self, request, response):
        response.patch = self.patch
        response.mask = self.mask
        response.image = self.image
        self.normal_msg.data = self.normal.flatten().tolist()
        response.normal = self.normal_msg

        return response
    
    def _pointcloud_init(self):
        '''
        Create a initial pointcloud with a flat surface, along with an image of the surface
        '''
        # Create a initial numpy pointcloud
        x = np.arange(self.imgw)
        y = np.arange(self.imgh)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X)

        self.points = np.zeros([self.imgw * self.imgh, 3])
        self.points[:, 0] = np.ndarray.flatten(X)  
        self.points[:, 1] = np.ndarray.flatten(Y)  
        self.points[:, 2] = self._depth2points(Z)

        # Create header
        header = Header()
        header.frame_id = 'gelsight_pointcloud'
        header.stamp = self.get_clock().now().to_msg()

        # Create pointcloud2 message
        self.patch = pc_utils.create_cloud(header=header, fields=FIELDS_XYZ, points=self.points)
        self.mask = pc_utils.create_cloud(header=header, fields=FIELDS_XYZ, points=self.points)
        

    def _update_pointcloud(self, Z, modality):
        '''
        Consantly update the pointcloud with the new heightmap image
        '''
        self.points[:, 2] = self._depth2points(Z)
        header = Header()
        header.frame_id = 'gelsight_patch'
        header.stamp = self.get_clock().now().to_msg()
        if modality == 'patch':
            self.patch = pc_utils.create_cloud(header=header, fields=FIELDS_XYZ, points=self.points)
        elif modality == 'mask':
            self.mask = pc_utils.create_cloud(header=header, fields=FIELDS_XYZ, points=self.points)

    def _depth2points(self, Z):
        return np.ndarray.flatten(Z)
    
    def _multiarray_init(self):
        self.normal_msg = Float32MultiArray()
        self.normal_msg.data = np.zeros([self.imgh, self.imgw, 2]).flatten().tolist()
        
        # Optionally set layout for multi-dimensional array (not strictly necessary here)
        self.normal_msg.layout.dim.append(MultiArrayDimension())
        self.normal_msg.layout.dim.append(MultiArrayDimension())
        self.normal_msg.layout.dim.append(MultiArrayDimension())

        self.normal_msg.layout.dim[0].label = 'height'
        self.normal_msg.layout.dim[0].size = self.imgh
        self.normal_msg.layout.dim[0].stride = self.imgh * self.imgw * 2
        
        self.normal_msg.layout.dim[1].label = 'width'
        self.normal_msg.layout.dim[1].size = self.imgw
        self.normal_msg.layout.dim[1].stride = self.imgw * 2
        
        self.normal_msg.layout.dim[2].label = 'channels'
        self.normal_msg.layout.dim[2].size = 2
        self.normal_msg.layout.dim[2].stride = 2


def main(args=None):
    rclpy.init(args=args)

    server = GelsightPointcloudServer()
    executor = MultiThreadedExecutor()
    executor.add_node(server)

    executor.spin()

    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()