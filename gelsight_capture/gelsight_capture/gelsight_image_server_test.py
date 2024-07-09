import cv2
from cv_bridge import CvBridge

from threading import Thread, Lock

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image, PointCloud2, PointField

import gsdevice

class WebcamVideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()

class GelsightTestServer(Node):
    def __init__(self):
        super().__init__('gelsight_tes_client')
        
        multithread_group = ReentrantCallbackGroup()
        self.cvbridge = CvBridge() # for converting ROS images to OpenCV images
        
        # the actual camera that captures the images
        self.camera = WebcamVideoStream(src=gsdevice.get_camera_id('GelSight Mini'))
        self.camera.start()
        
        self.imgh = 240
        self.imgw = 320
        
        # image publisher
        self.image_publisher = self.create_publisher(
            msg_type=Image, 
            topic='/gelsight/test_image', 
            qos_profile=10)
        image_timer_period = 0.05  # in seconds. equal to 20 Hz
        self.image_timer = self.create_timer(
            timer_period_sec=image_timer_period, 
            callback=self.img_timer_callback,
            callback_group=multithread_group)

        
    def img_timer_callback(self):
        original_image = self.camera.read()
        resized_image = cv2.resize(original_image, (self.imgw, self.imgh))
        ros_image = self.cvbridge.cv2_to_imgmsg(resized_image, encoding='bgr8')
        self.image_publisher.publish(ros_image)
        



def main(args=None):
    rclpy.init(args=args)

    server = GelsightTestServer()

    rclpy.spin(server)

    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()