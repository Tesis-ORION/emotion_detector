from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'emotion_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.xml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.keras')),
        (os.path.join('share', package_name, 'models'), glob('models/*.tflite')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Miguel Angel Gonzalez Rodriguez',
    maintainer_email='miguel_gonzalezr@ieee.org',
    description='Package that uses opencv and haarcascade to recognize facial emotions',
    license='BSD-3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "emotionDetector = emotion_detector.emotion:main",
            "emotionDetectorCoral = emotion_detector.emotion_coral:main",
            "camera = emotion_detector.camera:main",
        ],
    },
)
