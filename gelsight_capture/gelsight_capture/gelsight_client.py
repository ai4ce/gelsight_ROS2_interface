from gelsight_interface_msg.srv import TakePointcloud

from sensor_msgs.msg import Joy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc_utils

from queue import Queue

import os
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node



class GelsightClient(Node):

    def __init__(self):
        super().__init__('gelsight_client') # type: ignore


        ############################ Miscanellous Setup #######################################
        self._debounce_setup() # for debouncing the capture button
        self.shutter = False # when this is true, the client will issue a request to the server to capture pointcloud

        ############################ Launch Parameters ########################################
        # parameter handling
        self.declare_parameter(name = 'save_folder', value = '/home/irving/Desktop')
        self.save_folder = self.get_parameter('save_folder').get_parameter_value().string_value

        ############################ Client Setup #############################################
        # gelsight client
        self.gs_cli = self.create_client(
            srv_type=TakePointcloud, 
            srv_name='/gelsight/get_pointcloud')
        while not self.gs_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('RGB service not available, waiting again...')
        
        self.gs_req = TakePointcloud.Request()


        ############################ Subscriber Setup #########################################
        # subscribe to joy topic to read joystick button press
        self.joy_sub = self.create_subscription(
            msg_type=Joy, 
            topic='/joy', 
            callback=self.joy_callback, 
            qos_profile=10)

    def joy_callback(self, msg):
        # buttong 9 is the menu button
        old_value = self.debounce_buffer.get() # pop the oldest read
        self.debounce_buffer.put(msg.buttons[9]) # push the newest read
        if old_value == 0 and msg.buttons[9] == 1: 
            self.shutter = True # rising edge detected

    def postprocess(self, ros_cloud):
        '''
        img: sensor_msgs.msg.PointCloud2
        Convert a PointCloud2 message to a pcd and save it to disk
        '''
        
        # Get cloud data from ros_cloud
        field_names=[field.name for field in ros_cloud.fields]
        cloud_data = list(pc_utils.read_points(ros_cloud, skip_nans=True, field_names = field_names))
        
        # Check empty
        open3d_cloud = o3d.geometry.PointCloud()
        if len(cloud_data)==0:
            self.get_logger().warn('Empty pointcloud, not saving.')
            return
        
        # populate the open3d cloud
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

        # save the pointcloud
        o3d.io.write_point_cloud(
            os.path.join(self.save_folder, f'pc_{ros_cloud.header.stamp.sec}.pcd'),
            open3d_cloud
        )
        # color the log message
        color_start = '\033[94m'
        color_reset = '\033[0m'

        self.get_logger().info(f'{color_start} Pointcloud saved.{color_reset}')
    
    def _debounce_setup(self):
        '''
        As in any embedded system, we need to debounce the capture button.
        While we human think we press the button once, the computer actually consider the button pressed all the time during the duration of the press,
        because the polling rate is much faster than the human reaction time. 
        
        This function sets up a buffer to store the last value of the button press so that we can detect the rising edge.
        '''

        self.debounce_buffer = Queue(maxsize=1)
        self.debounce_buffer.put(0) # when nothing is pressed, the value is 0

def main(args=None):
    rclpy.init(args=args)

    client = GelsightClient()
    while rclpy.ok():
        if client.shutter: # shutter down
            
            # Not exactly the most performant way to do things, because we are sorta making calls sequentially
            # But it's good enough for now

            # send request to server to capture images
            pointcloud_future = client.gs_cli.call_async(client.gs_req)

            # immediately shutter up to debounce, so we don't caputre multiple images
            client.shutter = False
            
            client.get_logger().info('Request to capture gelsight pointcloud sent...')
            
            # wait for the server to capture images
            rclpy.spin_until_future_complete(client, pointcloud_future)

            # get the images from the server
            client.get_logger().info('Pointcloud Acquired...')
            pointcloud_response = pointcloud_future.result()

            # postprocess the images
            client.postprocess(pointcloud_response.pointcloud)

        rclpy.spin_once(client)

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()