import os
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from object_detection_interfaces.msg import DetectionArray, Detection

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

def load_model(checkpoint_path, image_size, device='cuda'):
    model = fasterrcnn_resnet50_fpn(pretrained=False, max_size=image_size, min_size=image_size, num_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

class BBoxPredictor(Node):
    def __init__(self):
        super().__init__('bbox_predictor')
        # Subscribe to Image messages from /image_raw
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher for DetectionArray messages on the "/detections" topic
        self.detection_publisher = self.create_publisher(DetectionArray, '/detections', 10)
        
        self.bridge = CvBridge()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load the model into self.model
        # Change the path and image_size as needed for your model
        model_path = "/home/smoothie/Desktop/16280/16280-Student-Object-Detection-main/src/image_processing_pkg/model.pth"
        image_size = 800  # Change this to the image size your model uses!
        self.model = load_model(model_path, image_size, device=str(self.device))
        
        # Log that the predictor node was initialized
        self.get_logger().info("BBoxPredictor node initialized.")

    def image_callback(self, msg):
        # Log that an image was received for prediction
        self.get_logger().info("Received image for prediction.")
        
        # Convert ROS2 Image to OpenCV image using "bgr8" encoding
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error("Error converting image: " + str(e))
            return
        
        # Convert the OpenCV image to a tensor and prepare it for the model.
        image_tensor = torch.from_numpy(cv_image).permute(2, 0, 1).float()
        image_tensor = image_tensor.to(self.device)
        images = [image_tensor]

        # Run the model inference without gradient computation.
        with torch.no_grad():
            predictions = self.model(images)
        
        detection_msg = DetectionArray()
        # Sync the header timestamp with the input image
        detection_msg.header.stamp = msg.header.stamp  
        
        detection_count = 0
        # Iterate over the predictions and only include detections with confidence above 0.9
        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
            if score < 0.9:
                continue
            detection = Detection()
            detection.bbox = [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
            detection.label = int(label.item())
            detection.score = float(score.item())
            detection_msg.detections.append(detection)
            detection_count += 1

        # Publish the detection message
        self.detection_publisher.publish(detection_msg)
        # Log how many detections were published
        self.get_logger().info(f"Published {detection_count} detections.")

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    # Create the BBoxPredictor node
    node = BBoxPredictor()
    # Keep the node running to listen for messages
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Cleanup and shutdown ROS2
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
