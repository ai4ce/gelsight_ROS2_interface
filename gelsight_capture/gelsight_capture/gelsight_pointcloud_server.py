import cv2
from cv_bridge import CvBridge

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc_utils

from gelsight_interface_msg.srv import TakePointcloud

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

        ############################ Node Components Setup ########################################
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            self.imgh = config["h"]
            self.imgw = config["w"]
            self.mmpp = config["mmpp"]
        multithread_group = ReentrantCallbackGroup()
        self.cvbridge = CvBridge() # for converting ROS images to OpenCV images
        

        # 3d reconstructioner 
        self.recon = Reconstruction3D(resource_dir=resource_path)
    
        # Point cloud visualizer
        # self.viz = Visualize3D(self.imgh, self.imgw)

        self._pointcloud_init()
        
        # pointcloud publisher
        self.pointcloud_publisher = self.create_publisher(
            msg_type=PointCloud2, 
            topic='/gelsight_capture/pointcloud', 
            qos_profile=10)
        
        self.image_subscriber = self.create_subscription(
            msg_type=Image, 
            topic='/gelsight_capture/image', 
            callback=self.image_sub_callback,
            qos_profile=10,
            callback_group=multithread_group)
        
        self.pointcloud_service = self.create_service(
            srv_type=TakePointcloud, 
            srv_name='/gelsight_capture/get_pointcloud', 
            callback=self.get_pointcloud_callback,
            callback_group=multithread_group)
        # pointcloud_timer_period = 0.05  # in seconds. equal to 20 Hz
        # self.image_timer = self.create_timer(
        #     timer_period_sec=pointcloud_timer_period, 
        #     callback=self.img_timer_callback,
        #     callback_group=multithread_group)
    
        
    def image_sub_callback(self, msg: Image) -> None:
        original_image = self.cvbridge.imgmsg_to_cv2(img_msg=msg, desired_encoding='passthrough')
        gradient, heightmap, mask = self.recon.get_surface_info(original_image, self.mmpp)
        self._update_pointcloud(heightmap)
        self.pointcloud_publisher.publish(self.pointcloud)
    
    def get_pointcloud_callback(self, request, response):
        response.pointcloud = self.pointcloud
        return response
    
    def _pointcloud_init(self):
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
        self.pointcloud = pc_utils.create_cloud(header=header, fields=FIELDS_XYZ, points=self.points)

    def _update_pointcloud(self, Z):
        self.points[:, 2] = self._depth2points(Z)
        header = Header()
        header.frame_id = 'gelsight_pointcloud'
        header.stamp = self.get_clock().now().to_msg()
        self.pointcloud = pc_utils.create_cloud(header=header, fields=FIELDS_XYZ, points=self.points)

    def _depth2points(self, Z):
        return np.ndarray.flatten(Z)


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