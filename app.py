from flask import Flask, render_template, Response
import cv2
import numpy as np
import threading
import tensorflow as tf

app = Flask(__name__)

cam = None  
lock = threading.Lock() 

# Load your TensorFlow model (assuming it's saved in models/segmentation_model.h5)
model = tf.keras.models.load_model('models/segmentation_model.h5')

def preprocess_image(frame):
    img = cv2.resize(frame, (224, 224))  # Resize to model's input size
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_segmentation(frame):
    img = preprocess_image(frame)
    prediction = model.predict(img)[0]  # Get the mask prediction

    # Resize the predicted mask back to original frame size
    mask = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))

    # Threshold the mask to convert probabilities to binary (0 or 1)
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Apply the mask as an overlay (you can use different colors for different acne classes)
    overlay = frame.copy()
    overlay[mask == 255] = (0, 255, 0)  # Green for detected areas

    # Blend the overlay with the original frame
    blended_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    return blended_frame

def generate_frames():
    global cam
    while True:
        if cam is None:
            continue  

        with lock:
            success, frame = cam.read()
            if not success:
                break

            # Perform segmentation and overlay the result
            frame = predict_segmentation(frame)

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
            cam = cv2.VideoCapture(0)  
    return "Camera started", 200

if __name__ == '__main__':
    app.run(debug=True)
