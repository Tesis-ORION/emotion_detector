#!/usr/bin/env python3

import os
import time  # ⏱️ Para limitar frecuencia de publicación

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

import cv2
import tensorflow as tf
import numpy as np


class EmotionRecognizerNode(Node):
    def __init__(self):
        super().__init__('emotion_recognizer')

        self.bridge = CvBridge()

        # Ruta al Haar cascade y al modelo .keras
        pkg_share = get_package_share_directory('emotion_detector')
        cascade_path = os.path.join(pkg_share, 'models', 'haarcascade_frontalface_default.xml')
        model_path   = os.path.join(pkg_share, 'models', 'fer2013_mini_XCEPTION.208-0.61.keras')

        # Carga Haar y modelo
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.model = tf.keras.models.load_model(model_path)
        self.get_logger().info(f"✅ Modelo cargado desde: {model_path}")

        # Labels del modelo
        self.emotion_labels = [
            'Angry', 'Disgust', 'Fear',
            'Happy', 'Sad', 'Surprise', 'Neutral'
        ]

        # Mapeo personalizado a enteros
        self.emotion_to_int = {
            'Angry': 0,
            'Disgust': 1,
            'Fear': 2,
            'Happy': 3,
            'Neutral': 4,
            'Sad': 5,
            'Surprise': 6
        }

        # Publicadores
        self.pub_emotion     = self.create_publisher(String, "/emotion", 10)
        self.pub_emotion_int = self.create_publisher(Int32,  "/emotion/int", 10)
        self.pub_debug       = self.create_publisher(Image,  "/emotion/image_debug", 10)

        # Control de tiempo para limitar publicación de /emotion/int
        self.last_emotion_int_time = time.time()

        # Suscripción a imagen
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            10
        )

        self.get_logger().info("🎯 Nodo de reconocimiento de emociones inicializado")

    def image_callback(self, msg: Image):
        try:
            # ROS → OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar rostro
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)    # batch
                roi = np.expand_dims(roi, axis=-1)   # channel

                preds = self.model.predict(roi, verbose=0)
                idx   = int(np.argmax(preds[0]))
                label = self.emotion_labels[idx]
                conf  = float(preds[0][idx])

                # Dibuja
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label}: {conf*100:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2
                )

                # Publicar emoción como string (siempre)
                out_str = String()
                out_str.data = label
                self.pub_emotion.publish(out_str)

                # Publicar emoción como entero solo si ha pasado al menos 2 segundos
                if label in self.emotion_to_int:
                    now = time.time()
                    if (now - self.last_emotion_int_time) >= 0.5:
                        out_int = Int32()
                        out_int.data = self.emotion_to_int[label]
                        self.pub_emotion_int.publish(out_int)
                        self.last_emotion_int_time = now

            # Imagen de depuración
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.pub_debug.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"❌ Error al procesar imagen: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = EmotionRecognizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
