# emotion detector
[![language](https://img.shields.io/badge/language-python-239120)](#)
[![OS](https://img.shields.io/badge/OS-Ubuntu_24.04-0078D4)](#)
[![CPU](https://img.shields.io/badge/CPU-x86%2C%20x64%2C%20ARM%2C%20ARM64-FF8C00)](#)
[![GitHub release](https://img.shields.io/badge/release-v1.0.0-4493f8)](#)
[![GitHub release date](https://img.shields.io/badge/release_date-february_2025-96981c)](#)
[![GitHub last commit](https://img.shields.io/badge/last_commit-february_2025-96981c)](#)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Tesis-ORION/emotion_detector)

⭐ Star us on GitHub — it motivates us a lot!

## Table of Contents
- [About](#-about)
- [Demostration](#-demostration)
- [How to Build](#-how-to-build)
- [License](#-license)

## 🚀 About

**emotion detector** is a package for ROS2 Jazzy that allow us to capture camera video, recognize face mesh using *HaarCascade* model, and detect facial emotions using *DeepFace*.

## 🎥 Demostration
https://github.com/user-attachments/assets/0afdabfb-ed9f-404f-8ce3-3e7e2e5c7a6a

## 📝 How to Build

To build the packages, follow these steps:

```shell
# First you need to clone the repository in your workspace
cd ~/ros2_ws/src
git clone https://github.com/Tesis-ORION/emotion_detector

# Now you need to install the dependencies
cd emotion_detector/resources
pip install -r requirements --break-system-packages

# Next you need to compile the package and launch the project
cd ~/ros2_ws
colcon build --packages-select emotion_detector
ros2 launch emotion_detector emotionDetector.launch.py
```

## 📃 License

emotion_detector is available under the BSD-3-Clause license. See the LICENSE file for more details.
