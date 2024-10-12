from flask import Flask, render_template, Response
import cv2
import threading

app = Flask(__name__)

cam = None  
lock = threading.Lock() 


def generate_frames():
    global cam
    while True:
        if cam is None:
            continue  

        with lock:
            success, frame = cam.read()
            if not success:
                break
            else:
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
