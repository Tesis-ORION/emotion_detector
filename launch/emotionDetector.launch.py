#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    package_name = 'emotion_detector'
    rviz_config_path = os.path.join(get_package_share_directory(package_name), 'rviz', 'emotion.rviz')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz_config',
            default_value=rviz_config_path,
            description='Ruta al archivo de configuración de RViz'
        ),
        
        Node(
            package='emotion_detector',
            executable='emotionDetector'
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')]
        ),
    ])