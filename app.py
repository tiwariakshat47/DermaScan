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

def generate_frames():
    global cam, last_prediction
    while True:
        if cam is None:
            continue  

        with lock:
            success, frame = cam.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)

            # Perform prediction on each frame
            predicted_class_index, confidence = predict_class(frame)
            predicted_class_name = class_names[predicted_class_index]

            # Store the latest prediction
            last_prediction = {
                "class": predicted_class_name,
                "confidence": confidence
            }

            # Draw bounding box and display confidence level
            height, width, _ = frame.shape
            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (255, 0, 0), 2)
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
    if 'confidence' in last_prediction:
        last_prediction['confidence'] = float(last_prediction['confidence'])

    return jsonify({'last_prediction': last_prediction})

if __name__ == '__main__':
    app.run(debug=True)