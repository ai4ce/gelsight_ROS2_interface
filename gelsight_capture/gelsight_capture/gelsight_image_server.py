import cv2
from cv_bridge import CvBridge

import yaml

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image

from .gsdevice import Camera2D


class GelsightImageServer(Node):
    def __init__(self):
        super().__init__('gelsight_image_client') # type: ignore
        
        # it's complicated why we need to load the config path instead of the content of config. See launch file for explanation
        self.declare_parameter(name = 'config_path', value = '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            self.imgh = config["h"]
            self.imgw = config["w"]

        multithread_group = ReentrantCallbackGroup()
        self.cvbridge = CvBridge() # for converting ROS images to OpenCV images
        
        # the actual camera that captures the images
        self.camera2d = Camera2D(dev_type="GelSight Mini", 
                                 imgh=self.imgh, 
                                 imgw=self.imgw)
        self.camera2d.start()

        
        # image publisher
        self.image_publisher = self.create_publisher(
            msg_type=Image, 
            topic='/gelsight/image', 
            qos_profile=10)
        image_timer_period = 0.05  # in seconds. equal to 20 Hz
        self.image_timer = self.create_timer(
            timer_period_sec=image_timer_period, 
            callback=self.image_timer_callback,
            callback_group=multithread_group)

        
    def image_timer_callback(self):
        original_image = self.camera2d.read()
        # self.get_logger().info(f'image type: {type(original_image)}')
        ros_image = self.cvbridge.cv2_to_imgmsg(original_image, encoding='bgr8')
        self.image_publisher.publish(ros_image)
        



def main(args=None):
    rclpy.init(args=args)

    server = GelsightImageServer()
    executor = MultiThreadedExecutor()
    executor.add_node(server)

    executor.spin()

    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()