
import os
from flask import g
import yaml

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)

from launch_ros.actions import Node


from ament_index_python.packages import get_package_share_directory

from launch_xml.launch_description_sources import XMLLaunchDescriptionSource



def declare_arguments():
    return LaunchDescription(
        [
            DeclareLaunchArgument("save_folder", default_value="/home/irving/Desktop", description="What folder to save the images to"),
            DeclareLaunchArgument("config_name", default_value="gsmini", description="Name of the configuration file"),
            DeclareLaunchArgument('final_config_path', default_value=[get_package_share_directory("gelsight_capture"), '/config/', LaunchConfiguration('config_name'), '.yaml'])
        ]
    )



def load_yaml(package_name, file_name):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_name)

    # try:
    with open(absolute_file_path) as file:
        return yaml.safe_load(file)
    # except OSError:  # parent of IOError, OSError *and* WindowsError where available
    #     return None
    
def generate_launch_description():

    # ROS2 launch parameters cannot be directly access in the launch file, at least there is no easy way to do so.
    # Therefore, we need to load the yaml file path here and pass it as a parameter to the node using a dummy launch parameter.
    final_config_path = LaunchConfiguration("final_config_path")
    gelsight_resource_path = os.path.join(get_package_share_directory("gelsight_capture"), "gelsight_resources")
    save_folder_path = LaunchConfiguration("save_folder")

    ld = LaunchDescription()
    ld.add_entity(declare_arguments())
    
    foxglove_launch = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
                os.path.join(get_package_share_directory('foxglove_bridge'), 
                                    'launch', 
                                    'foxglove_bridge_launch.xml')
        )
    )

    
    joy_launch = Node(
                executable='joy_node',
                package='joy',
                name='joy_node',
                parameters=[
                    # {'autorepeat_rate': 50.0},
                ],
            )
    image_server_launch = Node(
        package='gelsight_capture',
        executable='gelsight_image_server',
        name='gelsight_image_server',
        parameters=[{'config_path': final_config_path}],
    )
    pointcloud_server_launch = Node(
        package='gelsight_capture',
        executable='gelsight_pointcloud_server',
        name='gelsight_pointcloud_server',
        parameters=[{'config_path': final_config_path, 'resource_path': gelsight_resource_path}],
    )
    
    gelsight_client_launch = Node(
        package='gelsight_capture',
        executable='gelsight_client',
        name='gelsight_client',
        parameters=[{'save_folder': save_folder_path}],
    )
    ld.add_action(foxglove_launch)
    ld.add_action(joy_launch)
    ld.add_action(image_server_launch)
    ld.add_action(pointcloud_server_launch)
    ld.add_action(gelsight_client_launch)

    return ld