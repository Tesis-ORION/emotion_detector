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
model = os.path.join(package_share_directory, 'models', "haarcascade_frontalface_default.xml")

class EmotionPublisher(Node):
    def __init__(self):
        super().__init__('emotion_recognizer')
        
        self.bridge = CvBridge()
        
        self.publisher_emotion_ = self.create_publisher(String, "/emotion", 10)
        self.get_logger().info("Emotion publisher started")
        
        self.publisher_camera_ = self.create_publisher(Image, "/camera/image_raw", 10)
        self.get_logger().info("Camera publisher started")
    
    def publishEmotion(self, emotion):
        msg = String()
        msg.data = emotion
        self.publisher_emotion_.publish(msg)
        self.get_logger().info("Published: %s" % msg.data)
        
    def publishCamera(self, video):
        ros_image = self.bridge.cv2_to_imgmsg(video, encoding='bgr8')
        self.publisher_camera_.publish(ros_image)
        
class EmotionRecognizer():
    def __init__(self):
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(model)

        # Start capturing video
        self.cap = 0
        
        self.node = EmotionPublisher()

    def camera(self):
        while True:
            self.cap = cv2.VideoCapture(0)
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion and its confidence score
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion] / 100

                # Draw rectangle around face and label with predicted emotion and confidence
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"{emotion}: {confidence*100:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                self.node.publishEmotion(emotion)
                self.node.publishCamera(frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Release the capture and close all windows
            self.cap.release()
            cv2.destroyAllWindows()

def main(args = None):
    rclpy.init(args=args)
    
    recognizer = EmotionRecognizer()
    
    recognizer.camera()
    
    rclpy.shutdown()
        
        
if __name__ == "__main__":
    main()