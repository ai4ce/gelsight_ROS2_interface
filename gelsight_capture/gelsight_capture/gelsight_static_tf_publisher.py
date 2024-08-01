import yaml

from geometry_msgs.msg import TransformStamped

import rclpy
from rclpy.node import Node

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

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

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Publish static transforms once at startup
        self.make_transforms()

    def make_transforms(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_eef'
        t.child_frame_id = 'link_gelsight'

        # calculate the x distance between the end effector and the gelsight in meters
        z_distance = (self.mount_length + self.gs_length + self.gel_thickness) / 1000

        t.transform.translation.x = float(0)
        t.transform.translation.y = float(0)
        t.transform.translation.z = float(z_distance)

        # R is a 3x3 identity rotation matrix
        # R = np.eye(3)
        # quat = mat2quat(R)
        # self.get_logger().info(f"quat: {quat}")
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