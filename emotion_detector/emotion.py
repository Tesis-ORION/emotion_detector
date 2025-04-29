import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from ament_index_python.packages import get_package_share_directory

import cv2
from deepface import DeepFace
import os

package_share_directory = get_package_share_directory("emotion_detector")
model_path = os.path.join(package_share_directory, 'models', "haarcascade_frontalface_default.xml")


class EmotionRecognizerNode(Node):
    def __init__(self):
        super().__init__('emotion_recognizer')

        # Cargar el clasificador Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(model_path)

        # Publicador de emociones
        self.publisher_emotion = self.create_publisher(Int32, "/emotion", 10)

        # Cámara normal (webcam por defecto)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("❌ No se pudo abrir la cámara")
            return

        self.get_logger().info("🎯 Nodo de reconocimiento de emociones inicializado con cámara local")

        # Procesar en bucle
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ No se pudo capturar un frame de la cámara")
            return

        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]

                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion] / 100

                # Dibujar en la imagen
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"{emotion}: {confidence*100:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Publicar emoción
                msg_out = Int32()
                if emotion == "happy":
                    msg_out.data = 0
                elif emotion == "neutral":
                    msg_out.data = 1
                elif emotion == "sad":
                    msg_out.data = 2
                elif emotion == "angry":
                    msg_out.data = 3
                    
                
                self.publisher_emotion.publish(msg_out)
                self.get_logger().info(f"📤 Emoción publicada: {emotion}")

            # Mostrar imagen en una ventana (opcional)
            cv2.imshow("Emotion Recognition", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"❌ Error procesando frame: {e}")

    def destroy_node(self):
        super().destroy_node()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = EmotionRecognizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
