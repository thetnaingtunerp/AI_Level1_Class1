from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import download_model, load_model, run_inference, draw_boxes
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable for model
model = None

@app.before_first_request
def initialize_model():
    """Initialize the TensorFlow model before first request"""
    global model
    try:
        model = load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Handle image upload and object detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file:
        try:
            # Read and preprocess image
            image = Image.open(file.stream)
            image_np = np.array(image)
            
            # Convert RGBA to RGB if necessary
            if image_np.shape[-1] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            # Run object detection
            if model is None:
                initialize_model()
            
            output_dict = run_inference(model, image_np)
            
            # Draw bounding boxes
            image_with_boxes = draw_boxes(
                image_np, 
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                output_dict['detection_class_names']
            )
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            result_image = Image.fromarray(image_with_boxes)
            result_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare detection results
            detections = []
            for i in range(len(output_dict['detection_scores'])):
                if output_dict['detection_scores'][i] > 0.5:  # Confidence threshold
                    detections.append({
                        'class': output_dict['detection_class_names'][i],
                        'score': float(output_dict['detection_scores'][i]),
                        'box': output_dict['detection_boxes'][i].tolist()
                    })
            
            return jsonify({
                'success': True,
                'image': f"data:image/jpeg;base64,{img_str}",
                'detections': detections,
                'count': len(detections)
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/webcam')
def webcam_detection():
    """Page for webcam-based object detection"""
    return render_template('webcam.html')

if __name__ == '__main__':
    # Download model if not exists
    download_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)