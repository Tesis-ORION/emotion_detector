#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    package_name = 'emotion_detector'
    pkg_share = get_package_share_directory(package_name)
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'emotion.rviz')

    # Argumento para elegir el modo de detección: normal o coral
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='normal',
        description="Mode of emotion detection: 'normal' or 'coral'"
    )

    # Argumento para habilitar/deshabilitar RViz (por defecto sí)
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to launch RViz (true/false)'
    )

    # Argumento para la ruta del archivo de configuración de RViz
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=rviz_config_path,
        description='Path to RViz config file'
    )

    # Nodo normal
    normal_node = Node(
        package=package_name,
        executable='emotionDetector',
        name='emotion_detector_normal',
        condition=UnlessCondition(
            PythonExpression(["'", LaunchConfiguration('mode'), "' == 'coral'"])
        ),
        output='screen'
    )

    # Nodo para Coral
    coral_node = Node(
        package=package_name,
        executable='emotionDetectorCoral',
        name='emotion_detector_coral',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('mode'), "' == 'coral'"])
        ),
        output='screen'
    )

    # Nodo RViz solo si se habilita
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        mode_arg,
        rviz_arg,
        rviz_config_arg,
        normal_node,
        coral_node,
        rviz_node
    ])
