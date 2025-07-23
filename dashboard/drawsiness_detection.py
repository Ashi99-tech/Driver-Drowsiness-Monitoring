import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Drowsiness parameters
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15
frame_count = 0

def calculate_ear(eye_landmarks):
    vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def generate_frames():
    global frame_count
    cap = cv2.VideoCapture(0)  # Open webcam
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [np.array([face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]) 
                            for idx in [33, 160, 158, 133, 153, 144]]
                right_eye = [np.array([face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]) 
                             for idx in [362, 385, 387, 263, 373, 380]]
                
                h, w, _ = frame.shape
                left_eye = np.array([(int(x * w), int(y * h)) for x, y in left_eye])
                right_eye = np.array([(int(x * w), int(y * h)) for x, y in right_eye])
                
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESHOLD:
                    frame_count += 1
                    if frame_count >= CONSECUTIVE_FRAMES:
                        cv2.putText(frame, "Drowsiness Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1.5, (0, 0, 255), 3)
                else:
                    frame_count = 0

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
