from flask import Flask, render_template, Response
import cv2
import numpy as np
import threading
import tensorflow as tf

app = Flask(__name__)

cam = None  
lock = threading.Lock()

# Load your TensorFlow model (adjust the path to your model if needed)
model = tf.keras.models.load_model('models/segmentation_model.h5')

# Define class names (adjust this if you're not using CIFAR-10 classes)
class_names = ['blackheads', 'dark spot', 'nodules', 'papules', 'pustules', 'whiteheads']

# Update input size based on your model's requirements
input_size = (62, 62)  # The model expects images of size 62x62 with 3 channels

def preprocess_image(frame):
    img = cv2.resize(frame, input_size)  # Resize to (62, 62)
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_class(frame):
    img = preprocess_image(frame)

    # Make prediction
    prediction = model.predict(img)

    # Get the predicted class index and confidence
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)  # Get the confidence level

    return predicted_class_index, confidence

def generate_frames():
    global cam
    while True:
        if cam is None:
            continue  

        with lock:
            success, frame = cam.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)

            # Perform prediction
            predicted_class_index, confidence = predict_class(frame)
            predicted_class_name = class_names[predicted_class_index]

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global cam
    with lock:
        if cam is None:
            cam = cv2.VideoCapture(0)  # Start the webcam
    return "Camera started", 200

if __name__ == '__main__':
    app.run(debug=True)
