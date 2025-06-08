# emotion detector
[![language](https://img.shields.io/badge/language-python-239120)](#)
[![OS](https://img.shields.io/badge/OS-Ubuntu_24.04-0078D4)](#)
[![ROS](https://img.shields.io/badge/ROS_Version-Jazzy_Jalisco-0078D4)](#)
[![CPU](https://img.shields.io/badge/CPU-x86%2C%20x64%2C%20ARM%2C%20ARM64-FF8C00)](#)
[![GitHub release](https://img.shields.io/badge/release-v2.0.0-4493f8)](#)
[![GitHub release date](https://img.shields.io/badge/release_date-february_2025-96981c)](#)
[![GitHub last commit](https://img.shields.io/badge/last_commit-june_2025-96981c)](#)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Tesis-ORION/emotion_detector)

⭐ Star us on GitHub — it motivates us a lot!

## Table of Contents
- [About](#-about)
- [Demostration](#-demostration)
- [Train Model](#-train-model)
- [How to Build](#-how-to-build)
- [License](#-license)

## 🚀 About

This repository contains a ROS 2 package for real-time facial emotion recognition using the FER-2013 dataset, optimized for Edge TPU acceleration on Coral devices. It leverages a quantized Mini-XCEPTION architecture trained on FER-2013 and compiled with TensorFlow Lite. The node subscribes to a camera feed, detects faces using Haar cascades, classifies emotions, and publishes both textual and numerical labels along with annotated images. Ideal for integrating AI-based emotion detection in robotics, human-robot interaction, and embedded systems.

## 🎥 Demostration
https://github.com/user-attachments/assets/0afdabfb-ed9f-404f-8ce3-3e7e2e5c7a6a

## 🧠 Train Model
To train the model (if you want to because a default model is in `train/trained_models/emotion_models/`) you need to clone the repo and install the dependencies:
```
cd ~/ros2_ws/src
git clone https://github.com/Tesis-ORION/emotion_detector
cd emotion_detector/train
pip install -r requirements.txt --break-system-packages
```

Now you need to download the dataset, in this case we are going to use <a href="https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition">FER2013</a>, so you need to go to the link, download the zip in `emotion_detector/train/datasets` and unzip the dataset.

After you downloaded the dataset you can run the train code:
```
cd ~/ros2_ws/src/emotion_detector/train/src
chmod +x train_emotion_classifier.py
./train_emotion_classifier.py
```
This will generate a `.keras` model and the respective `accuracy and loss` graph and `ROC curves` for each emotion.

If you want to convert the model to tflite to use it with Google coral USB Accelerator you can run the code to convert the model:
```
cd ~/ros2_ws/src/emotion_detector/train
# In convert_to_tflite.py, change model name to the one you generated in emotion_detector/train/trained_models/emotion_models
chmod +x convert_to_tflite.py
./convert_to_tflite.py
edgetpu_compiler model_name.tflite
```
Now you can move the `.keras` or `.tflite` coral compatible model to `emotion_detector/models`

DISCLAIMER: If you want to use your own model just put in in `emotion_detector/models` but make sure to change the model name in the `emotion.py` or `emotion_coral.py` node

## 📝 How to Build

To build the packages (only if you are using ROS 2 Jazzy and python 3.12), follow these steps:

```shell
# First you need to clone the repository in your workspace
cd ~/ros2_ws/src
git clone https://github.com/Tesis-ORION/emotion_detector

# Now you need to install the dependencies
cd emotion_detector/resources
pip install -r requirements --break-system-packages
# For Google Coral USB Accelerator support
pip install tflite_runtime-2.17.1-cp312-cp312-linux_aarch64.whl pycoral-2.0.3-cp312-cp312-linux_aarch64.whl --break-system-packages

# Next you need to compile the package
cd ~/ros2_ws
colcon build --packages-select emotion_detector
source install/setup.bash
```

First make sure you are running the respective camera node. If you are using a normal camera you can use:
```
$ ros2 run emotion_detector camera
```
and check that in `emotion.py` (Line 64) or `emotion_coral.py` (Line 69) node you are using the correct camera topic.


If you want to launch the package without `Google Coral USB Accelerator` support use:
```
ros2 launch emotion_detector emotionDetector.launch.py
```

If you want to launch the package using `Google Coral USB Accelerator` support use:
```
ros2 launch emotion_detector emotionDetector.launch.py mode:=coral
```

If you want to launch the package without using `rviz` use:
```
ros2 launch emotion_detector emotionDetector.launch.py rviz:=false
```

## 📃 License

emotion_detector is available under the BSD-3-Clause license. See the LICENSE file for more details.
