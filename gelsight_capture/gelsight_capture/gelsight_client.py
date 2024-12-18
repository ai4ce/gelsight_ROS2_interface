from gelsight_interface_msg.srv import GSCapture

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Joy
import sensor_msgs_py.point_cloud2 as pc_utils

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from queue import Queue
import os
import json
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

class GelsightClient(Node):

    def __init__(self):
        super().__init__('gelsight_client') # type: ignore

        ############################ Launch Parameters ########################################
        # parameter handling
        self.declare_parameter(name = 'save_folder', value = '/home/irving/Desktop')
        self.save_folder = self.get_parameter('save_folder').get_parameter_value().string_value
        self.declare_parameter(name = 'json_path', value = '')
        self.json_path = self.get_parameter('json_path').get_parameter_value().string_value

        ############################ Miscanellous Setup #######################################
        self._debounce_setup() # for debouncing the capture button
        self.shutter = False # when this is true, the client will issue a request to the server to capture pointcloud

        # create a folder to save the tactile info
        # save_folder/tactile/patch + save_folder/tactile/mask + save_folder/tactile/image + save_folder/tactile/normal
        if self.save_folder != '':
            self._folder_init()

        self.pointcloud_count = 0

        self.cvbridge = CvBridge()
        ############################ JSON Setup ###############################################
        if self.json_path != '':
            self.json_dict = {}
            self._json_setup()

        ############################ TF Setup #################################################
        # buffer to hold the transform in a cache
        self.tf_buffer = Buffer()

        # listener. Important to spin a thread, otherwise the listen will block and no TF can be updated
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self, spin_thread=True)

        ############################ Client Setup #############################################
        # gelsight client
        self.gs_cli = self.create_client(
            srv_type=GSCapture, 
            srv_name='/gelsight_capture/capture')
        while not self.gs_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('RGB service not available, waiting again...')
        
        self.gs_req = GSCapture.Request()

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

    def postprocess(self, response):
        '''
        response: gelsight_interface_msg.srv.GSCapture.Response
        Convert a PointCloud2 message to a pcd and save it to disk
        '''
        
        patch_cloud = self._roscloud2o3d(response.patch)
        mask_cloud = self._roscloud2o3d(response.mask)
        image = self.cvbridge.imgmsg_to_cv2(response.image, desired_encoding='passthrough')
        normal = self._rosarray2np(response.normal)

        # color the log message in yellow
        color_start = '\033[93m'
        color_reset = '\033[0m'

        if self.save_folder != '':
            self._save_data(patch_cloud, mask_cloud, image, normal)

            self.get_logger().info(f'{color_start}Data saved.{color_reset}')
        else:
            self.get_logger().info(f'{color_start}Data captured, not saved{color_reset}')
        
        # update the json dict
        if self.json_path != '':
            self._json_update()

            # overwrite the JSON file if there is one. Not sure if this is the best way to do it
            # potentially we can just keep the json_dict in memory and dump it at the end of the program
            # but this is a good way to keep the json file updated in case the program cannot exit as expected
            with open(self.json_path, 'wt') as f:
                json.dump(self.json_dict, f)
            
            # color the log message in green
            color_start = '\033[92m'
            color_reset = '\033[0m'
            self.get_logger().info(f'{color_start}JSON file updated{color_reset}')

        self.pointcloud_count += 1

    def _debounce_setup(self):
        '''
        As in any embedded system, we need to debounce the capture button.
        While we human think we press the button once, the computer actually consider the button pressed all the time during the duration of the press,
        because the polling rate is much faster than the human reaction time. 
        
        This function sets up a buffer to store the last value of the button press so that we can detect the rising edge.
        '''

        self.debounce_buffer = Queue(maxsize=1)
        self.debounce_buffer.put(0) # when nothing is pressed, the value is 0

    def _json_setup(self):
        '''
        Set up the json we are about to dump.
        '''
        # with open(calibration_path, "r") as f:
        #     config = yaml.safe_load(f)
        #     self.json_dict['w'] = config['w']
        #     self.json_dict['h'] = config['h']
        #     self.json_dict['fl_x'] = config['fl_x']
        #     self.json_dict['fl_y'] = config['fl_y']
        #     self.json_dict['cx'] = config['cx']
        #     self.json_dict['cy'] = config['cy']
        #     self.json_dict['k1'] = config['k1']
        #     self.json_dict['k2'] = config['k2']
        #     self.json_dict['p1'] = config['p1']
        #     self.json_dict['p2'] = config['p2']
        #     self.json_dict['camera_model'] = 'OPENCV'
        self.json_dict['frames'] = list()
        self.json_dict['applied_transform'] = np.array([[1.0, 0.0, 0.0, 0.0],
                                                        [0.0, 1.0, 0.0, 0.0],
                                                        [0.0, 0.0, 1.0, 0.0]], dtype=np.float64).tolist()

    def _json_update(self):
        '''
        Update the json dict with the latest transform
        '''
        update_dict = {}
        
        # get the coordinate of the camera in the base frame
        transformstamp = self.tf_buffer.lookup_transform(target_frame='link_base', 
                                            source_frame='link_gelsight', 
                                            time=Time(), 
                                            timeout=Duration(seconds=2))
        transformation_matrix = self._process_tf(transformstamp)


        update_dict['patch_path'] = os.path.join('tactile/patch', f'patch_{self.pointcloud_count}.pcd')
        update_dict['mask_path'] = os.path.join('tactile/mask', f'mask_{self.pointcloud_count}.pcd')
        update_dict['image_path'] = os.path.join('tactile/image', f'image_{self.pointcloud_count}.png')
        update_dict['normal_path'] = os.path.join('tactile/normal', f'normal_{self.pointcloud_count}.npy')

        update_dict['transform_matrix'] = transformation_matrix.tolist()
        update_dict['colmap_im_id'] = self.pointcloud_count

        self.json_dict['frames'].append(update_dict)
        
    def _process_tf(self, transformstamp):
        '''
        Turn the transformstamp into a 4x4 transformation matrix

        Input: geometry_msgs.msg.TransformStamped
        Output: np.array(4x4)
        '''
        translation = np.array([transformstamp.transform.translation.x, transformstamp.transform.translation.y, transformstamp.transform.translation.z])
        quaternion = np.array([transformstamp.transform.rotation.x, transformstamp.transform.rotation.y, transformstamp.transform.rotation.z, transformstamp.transform.rotation.w])
        
        # convert quaternion to rotation matrix with scipy, which I think is more trustworthy than transforms3d
        rotation = Rotation.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()

        # create the 4x4 transformation matrix
        transformation_matrix = np.eye(4, dtype=np.float64)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation

        return transformation_matrix
    
    def _roscloud2o3d(self, ros_cloud):
        '''
        Convert a PointCloud2 message to a pcd
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

        return open3d_cloud
    
    def _rosarray2np(self, ros_array):
        '''
        Convert a Float32MultiArray message to a numpy array
        '''
        return np.array(ros_array.data).reshape(ros_array.layout.dim[0].size, ros_array.layout.dim[1].size, ros_array.layout.dim[2].size)
    
    def _save_data(self, patch, mask, image, normal):
        '''
        Save the patch, mask, image, and normal to disk
        '''
        o3d.io.write_point_cloud(
            os.path.join(self.patch_save_folder, f'patch_{self.pointcloud_count}.pcd'),
            patch
        )
        o3d.io.write_point_cloud(
            os.path.join(self.patch_save_folder, f'patch_{self.pointcloud_count}.ply'),
            patch
        )
        
        o3d.io.write_point_cloud(
            os.path.join(self.mask_save_folder, f'mask_{self.pointcloud_count}.pcd'),
            mask
        )
        o3d.io.write_point_cloud(
            os.path.join(self.mask_save_folder, f'mask_{self.pointcloud_count}.ply'),
            mask
        )

        cv2.imwrite(os.path.join(self.image_save_folder, f'image_{self.pointcloud_count}.png'), image)
        np.save(os.path.join(self.normal_save_folder, f'normal_{self.pointcloud_count}.npy'), normal)

    def _folder_init(self):
        os.makedirs(self.save_folder, exist_ok=True)
        self.save_folder = os.path.join(self.save_folder, 'tactile')
        os.makedirs(self.save_folder, exist_ok=True)
        self.patch_save_folder = os.path.join(self.save_folder, 'patch')
        os.makedirs(self.patch_save_folder, exist_ok=True)
        self.mask_save_folder = os.path.join(self.save_folder, 'mask')
        os.makedirs(self.mask_save_folder, exist_ok=True)
        self.image_save_folder = os.path.join(self.save_folder, 'image')
        os.makedirs(self.image_save_folder, exist_ok=True)
        self.normal_save_folder = os.path.join(self.save_folder, 'normal')
        os.makedirs(self.normal_save_folder, exist_ok=True)

def main(args=None):
    rclpy.init(args=args)

    client = GelsightClient()
    while rclpy.ok():
        if client.shutter: # shutter down
            
            # send request to server to capture images
            pointcloud_future = client.gs_cli.call_async(client.gs_req)

            # immediately shutter up to debounce, so we don't caputre multiple pointclouds
            client.shutter = False
            
            client.get_logger().info('Request to capture gelsight pointcloud sent...')
            
            # wait for the server to capture
            rclpy.spin_until_future_complete(client, pointcloud_future)

            # get the images from the server
            client.get_logger().info('Pointcloud Acquired...')
            pointcloud_response = pointcloud_future.result()

            # postprocess the images
            client.postprocess(pointcloud_response)

        rclpy.spin_once(client)

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()