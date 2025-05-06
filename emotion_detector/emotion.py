import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

import cv2
from deepface import DeepFace
import os

package_share_directory = get_package_share_directory("emotion_detector")
model_path = os.path.join(package_share_directory, 'models', "haarcascade_frontalface_default.xml")


class EmotionRecognizerNode(Node):
    def __init__(self):
        super().__init__('emotion_recognizer')

        self.bridge = CvBridge()

        # Cargar el clasificador Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(model_path)

        # Publicadores
        self.publisher_emotion = self.create_publisher(String, "/emotion", 10)
        self.publisher_debug_image = self.create_publisher(Image, "/emotion/image_debug", 10)

        # Suscriptor al tópico de la cámara
        self.subscription = self.create_subscription(
            Image,
            "/apc/left/image_color",
            self.image_callback,
            10
        )

        self.get_logger().info("🎯 Nodo de reconocimiento de emociones inicializado y suscrito a /apc/left/image_color")

    def image_callback(self, msg):
        try:
            # Convertir imagen ROS a OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detectar rostros
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Analizar emoción
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion] / 100

                # Dibujar sobre la imagen
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"{emotion}: {confidence*100:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Publicar emoción
                msg_out = String()
                msg_out.data = emotion
                self.publisher_emotion.publish(msg_out)
                self.get_logger().info(f"📤 Emoción publicada: {emotion}")

            # Publicar imagen anotada
            debug_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_debug_image.publish(debug_image_msg)

        except Exception as e:
            self.get_logger().error(f"❌ Error procesando imagen: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = EmotionRecognizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
