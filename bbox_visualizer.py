import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from object_detection_interfaces.msg import DetectionArray
import message_filters  # For synchronizing messages

class BBoxVisualizer(Node):
    def __init__(self):
        super().__init__('bbox_visualizer')
        
        self.bridge = CvBridge()
        
        # Create message_filters subscribers for Image and DetectionArray messages.
        # The image topic is "/image_raw" and the detections topic is "/detections".
        self.image_sub = message_filters.Subscriber(self, Image, '/image_raw')
        self.detection_sub = message_filters.Subscriber(self, DetectionArray, '/detections')
        
        # Synchronize the two topics with an approximate time policy.
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.detection_sub],
                                                              queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        # Create a publisher that publishes Image messages to the topic "/image_with_bboxes"
        self.publisher = self.create_publisher(Image, '/image_with_bboxes', 10)

        # Define labels and corresponding colors for drawing
        self.labels_dict = {1: "blue", 2: "green", 3: "red"}
        self.label_colors = {
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "red": (0, 0, 255)
        }

        # Log that the BBox Visualizer node was successfully initialized
        self.get_logger().info("BBox Visualizer Node Initialized.")

    def callback(self, image_msg, detections_msg):
        # Convert the incoming Image message to an OpenCV image using "bgr8" encoding
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error("Error converting image: " + str(e))
            return
        
        # Draw bounding boxes and labels onto the image for each detection
        for detection in detections_msg.detections:
            x_min, y_min, x_max, y_max = map(int, detection.bbox)
            class_label = self.labels_dict.get(detection.label, "unknown")
            color = self.label_colors.get(class_label, (255, 255, 255))
            cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(cv_image, f"{class_label}: {detection.score:.2f}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Convert the processed OpenCV image back to a ROS2 Image message using "bgr8" encoding
        processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # Publish the processed image message
        self.publisher.publish(processed_msg)

        # Log that the image with bounding boxes has been published
        self.get_logger().info("Published image with bounding boxes.")

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    # Create the BBoxVisualizer node
    node = BBoxVisualizer()
    # Keep the node running
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Clean up and shutdown ROS2 properly
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
