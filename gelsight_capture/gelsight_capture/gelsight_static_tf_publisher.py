import yaml

from geometry_msgs.msg import TransformStamped

import rclpy
from rclpy.node import Node

from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class GelSightStaticTFPublisher(Node):
    """
    Broadcast transforms that never change.
    """

    def __init__(self):
        super().__init__('gelsight_static_tf_publisher') # type: ignore

        ############################ Launch Parameters ########################################

        # it's complicated why we need to load the config path instead of the content of config. See launch file for explanation
        self.declare_parameter(name = 'config_path', value = '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            self.mount_length = config['mount_length']
            self.gs_length = config['gs_length']
            self.gel_thickness = config['gel_thickness']
            self.x_offset = config['x_offset']
            self.y_offset = config['y_offset']

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        ############################# TF Setup ###############################################
        self.link_name = 'link_gelsight'
        # Publish static transforms once at startup
        self.make_transforms()

        # buffer to hold the transform in a cache
        self.tf_buffer = Buffer()

        # listener. Important to spin a thread, otherwise the listen will block and no TF can be updated
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self, spin_thread=True)

        self.pose_publisher = self.create_publisher(
        msg_type=TransformStamped, 
        topic='/gelsight_capture/gelsight_pose', 
        qos_profile=10)

        self.create_timer(0.5, self.publish_pose)
    
    def publish_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('link_base', self.link_name, Time(), timeout=Duration(seconds=2))
            self.pose_publisher.publish(t)
        except Exception as e:
            self.get_logger().info(f"Failed to publish pose: {e}")

    def make_transforms(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_eef'
        t.child_frame_id = self.link_name

        # calculate the distance between the end effector and the gelsight in meters
        x_distance = self.x_offset / 1000
        y_distance = self.y_offset / 1000
        z_distance = (self.mount_length + self.gs_length + self.gel_thickness) / 1000

        t.transform.translation.x = float(x_distance)
        t.transform.translation.y = float(y_distance)
        t.transform.translation.z = float(z_distance)

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0


        self.tf_static_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = GelSightStaticTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()