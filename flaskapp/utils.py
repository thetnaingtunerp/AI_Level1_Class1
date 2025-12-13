import tensorflow as tf
import numpy as np
import cv2
import os
import tarfile
import urllib.request
from PIL import Image

# Model configuration
MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
MODEL_DATE = '20200711'
MODEL_URL = f'http://download.tensorflow.org/models/object_detection/tf2/{MODEL_DATE}/{MODEL_NAME}.tar.gz'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME, 'saved_model')

# COCO labels
COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def download_model():
    """Download the pre-trained model if not exists"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Download tar file
        tar_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.tar.gz')
        urllib.request.urlretrieve(MODEL_URL, tar_path)
        
        # Extract tar file
        with tarfile.open(tar_path) as tar:
            tar.extractall(MODEL_DIR)
        
        # Remove tar file
        os.remove(tar_path)
        print("Model downloaded and extracted successfully!")

def load_model():
    """Load the TensorFlow model"""
    return tf.saved_model.load(MODEL_PATH)

def run_inference(model, image_np):
    """Run object detection on the image"""
    # The input needs to be a tensor, convert to uint8
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Run inference
    detections = model(input_tensor)
    
    # Convert to numpy arrays
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() 
                 for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # Convert detection_classes to integers and get class names
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    detections['detection_class_names'] = [
        COCO_LABELS[class_id] for class_id in detections['detection_classes']
    ]
    
    return detections

def draw_boxes(image_np, boxes, classes, scores, class_names, min_score=0.5):
    """Draw bounding boxes and labels on the image"""
    image_np = image_np.copy()
    height, width, _ = image_np.shape
    
    for i in range(len(scores)):
        if scores[i] >= min_score:
            # Get bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Create label
            label = f"{class_names[i]}: {scores[i]:.2f}"
            
            # Get text size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for text
            cv2.rectangle(image_np, 
                         (xmin, ymin - label_size[1] - 10),
                         (xmin + label_size[0], ymin),
                         color, -1)
            
            # Draw text
            cv2.putText(image_np, label, 
                       (xmin, ymin - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image_np