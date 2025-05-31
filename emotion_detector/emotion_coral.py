#!/usr/bin/env python3
"""
ROS2 node for real-time emotion recognition on Coral Edge TPU.
Loads a compiled TFLite model and processes camera images from a ROS topic.
Publishes detected emotion (label and int) and annotated image.
"""

import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input, input_size
from pycoral.adapters.classify import get_classes

class emotion_coral(Node):
    def __init__(self):
        super().__init__('emotion_recognizer_edge_tpu')
        self.bridge = CvBridge()

        # Paths
        pkg_share = get_package_share_directory('emotion_detector')
        cascade_path = os.path.join(pkg_share, 'models', 'haarcascade_frontalface_default.xml')
        tflite_path = os.path.join(pkg_share, 'models', 'mini_XCEPTION_int8_edgetpu.tflite')

        # Load Haar cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.get_logger().info(f"✅ Cascade loaded from: {cascade_path}")

        # Load Edge TPU interpreter
        self.interpreter = make_interpreter(tflite_path)
        self.interpreter.allocate_tensors()
        self.in_width, self.in_height = input_size(self.interpreter)
        self.get_logger().info(f"✅ Interpreter initialized with model: {tflite_path}")

        # Emotion labels
        self.emotion_labels = [
            'Disgust', 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        ]

        # Custom mapping to integer values
        self.emotion_to_int = {
            'Angry': 0,
            'Disgust': 1,
            'Fear': 2,
            'Happy': 3,
            'Neutral': 4,
            'Sad': 5,
            'Surprise': 6
        }

        # Publishers
        self.pub_emotion = self.create_publisher(String, '/emotion', 10)
        self.pub_emotion_int = self.create_publisher(Int32, '/emotion/int', 10)
        self.pub_debug   = self.create_publisher(Image, '/emotion/image_debug', 10)

        # Time control for throttling /emotion/int
        self.last_emotion_int_time = time.time()

        # Subscriber
        self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info('🎯 Edge TPU emotion recognizer initialized')

    def image_callback(self, msg: Image):
        try:
            # ROS Image -> OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )

            for (x, y, w, h) in faces:
                # Extract ROI and preprocess to model input size
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (self.in_width, self.in_height))
                roi = np.expand_dims(roi, axis=2)
                set_input(self.interpreter, roi)
                self.interpreter.invoke()

                # Get classification result
                classes = get_classes(self.interpreter, top_k=1, score_threshold=0.0)
                if not classes:
                    continue
                cls = classes[0]
                label = self.emotion_labels[cls.id]
                score = cls.score

                # Annotate frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label}: {score*100:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # Publish emotion as String
                msg_str = String()
                msg_str.data = label
                self.pub_emotion.publish(msg_str)

                # Publish emotion as Int32 (throttled to every 2 seconds)
                if label in self.emotion_to_int:
                    now = time.time()
                    if (now - self.last_emotion_int_time) >= 0.5:
                        msg_int = Int32()
                        msg_int.data = self.emotion_to_int[label]
                        self.pub_emotion_int.publish(msg_int)
                        self.last_emotion_int_time = now

            # Publish debug image
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.pub_debug.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"❌ Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = emotion_coral()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
