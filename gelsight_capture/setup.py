import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'gelsight_capture'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        # Include resource files
        (os.path.join('share', package_name, 'gelsight_resources'), glob(os.path.join('gelsight_resources', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Irving Fang',
    maintainer_email='irving.fang@nyu.edu',
    description='Interfaces the robot/joystick with the GelSight sensor.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gelsight_image_server = gelsight_capture.gelsight_image_server:main',
            'gelsight_pointcloud_server = gelsight_capture.gelsight_pointcloud_server:main',
            'gelsight_client = gelsight_capture.gelsight_client:main',
            'gelsight_static_tf_publisher = gelsight_capture.gelsight_static_tf_publisher:main'
        ],
    },
)
