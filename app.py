from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import threading
import tensorflow as tf
import openai
import os

app = Flask(__name__)

# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

cam = None  
lock = threading.Lock()
last_prediction = {"class": None, "confidence": 0}  # Store last prediction

# Load your TensorFlow model
model = tf.keras.models.load_model('models/segmentation_model.h5')

# Define class names
class_names = ['blackheads', 'dark spot', 'nodules', 'papules', 'pustules', 'whiteheads']

# Update input size based on your model's requirements
input_size = (62, 62)

def preprocess_image(frame):
    img = cv2.resize(frame, input_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_class(frame):
    img = preprocess_image(frame)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return predicted_class_index, confidence

def predict_class_and_box(frame):
    img = preprocess_image(frame)
    
    # Make a prediction (only class predictions)
    prediction = model.predict(img)
    
    # Print the raw prediction for debugging
    print(f"Raw prediction output: {prediction}")
    
    # Class prediction logic
    predicted_class_index = np.argmax(prediction[0])  # Class prediction
    confidence = np.max(prediction[0])  # Confidence for the class prediction
    
    # Print class index and confidence for debugging
    print(f"Predicted class index: {predicted_class_index}, Confidence: {confidence}")
    
    # Since the model does not predict bounding boxes, we can return a dummy box or skip it
    # Here we define a dummy bounding box that spans the entire image as a placeholder
    height, width = frame.shape[:2]
    predicted_box = [0.5, 0.5, 1.0, 1.0]  # Centered box, spans the entire image
    
    return predicted_class_index, confidence, predicted_box



import time  # Import time to track time intervals

# Keep track of the last time a bounding box was drawn and the class name
last_box_draw_time = 0
box_visible_duration = 1  # Duration for which the box should stay (in seconds)
last_box_info = {"class": None, "confidence": 0, "box": None}

def generate_frames():
    global cam, last_prediction, last_box_draw_time, last_box_info
    while True:
        if cam is None:
            continue

        with lock:
            success, frame = cam.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)

            # Perform prediction on each frame
            predicted_class_index, confidence, predicted_box = predict_class_and_box(frame)
            predicted_class_name = class_names[predicted_class_index]

            # Store the latest prediction
            last_prediction = {
                "class": predicted_class_name,
                "confidence": confidence
            }

            # Draw bounding box only if confidence is greater than 0.42
            current_time = time.time()
            if confidence > 0.42:
                # Store the time and box information when the prediction is above threshold
                last_box_draw_time = current_time
                last_box_info = {
                    "class": predicted_class_name,
                    "confidence": confidence,
                    "box": predicted_box
                }

            # Keep drawing the last bounding box if it's within the visible duration
            if current_time - last_box_draw_time < box_visible_duration:
                predicted_class_name = last_box_info["class"]
                confidence = last_box_info["confidence"]
                predicted_box = last_box_info["box"]

                # Draw bounding box based on last saved box coordinates
                height, width, _ = frame.shape
                x_center, y_center, box_width, box_height = predicted_box
                x_center_px = int(x_center * width)
                y_center_px = int(y_center * height)
                box_width_px = int(box_width * width)
                box_height_px = int(box_height * height)

                # Ensure bounding box coordinates are within the frame
                x1 = max(int(x_center_px - box_width_px / 2), 0)
                y1 = max(int(y_center_px - box_height_px / 2), 0)
                x2 = min(x1 + box_width_px, width)
                y2 = min(y1 + box_height_px, height)

                # Draw bounding box and add class/ confidence text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Predicted: {predicted_class_name}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f'Confidence: {confidence:.2f}', (15, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Encode the frame and send it to the client
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




def get_openai_response(predicted_class_name):
    prompt = f"What can you suggest for treating {predicted_class_name}?"
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a kind medical assistant providing skincare advice."},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content
        return reply
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Sorry, I couldn't provide advice at this moment."

@app.route('/')
def index():
    return render_template('index.html', pages = [['Home', 'index'],['Scan', 'scan']])

@app.route('/scan')
def scan():
    return render_template('scan.html', pages = [['Home', 'index'],['Scan', 'scan']])

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global cam
    with lock:
        if cam is None:
            cam = cv2.VideoCapture(0)
    return "Camera started", 200

@app.route('/get_advice')
def get_advice():
    if last_prediction["class"]:
        openai_response = get_openai_response(last_prediction["class"])
        return jsonify({'advice': openai_response})
    else:
        return jsonify({'advice': "No prediction available yet."})

# Helper function for frontend

@app.route('/get_prediction', methods=['GET'])
def get_last_prediction():
    return jsonify({'last_prediction': last_prediction})

if __name__ == '__main__':
    app.run(debug=True)