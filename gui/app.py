from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import serial
import time

app = Flask(__name__)

# Initialize serial communication with Arduino
try:
    arduino = serial.Serial('COM6', 9600, timeout=1)  # Change 'COM3' to your Arduino port
    time.sleep(2)  # Allow time for Arduino to initialize
    print("Connected to Arduino on COM3")
except Exception as e:
    print("Could not connect to Arduino:", e)
    arduino = None

# Function to send command to Arduino
def send_to_arduino(command):
    if arduino:
        try:
            arduino.write((command + '\n').encode())  # Send with newline
            time.sleep(0.1)  # Small delay to ensure Arduino reads data
            print("Sent to Arduino:", command)  # Debugging info
        except Exception as e:
            print("Error sending to Arduino:", e)

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks):
    vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Constants
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15
frame_count = 0

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
drowsy_status = "Monitoring..."

def generate_frames():
    global frame_count, drowsy_status
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        drowsy = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw facial landmarks
                
                left_eye = [np.array([face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]) 
                            for idx in [33, 160, 158, 133, 153, 144]]
                right_eye = [np.array([face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]) 
                             for idx in [362, 385, 387, 263, 373, 380]]
                
                h, w, _ = frame.shape
                left_eye = np.array([(int(x * w), int(y * h)) for x, y in left_eye])
                right_eye = np.array([(int(x * w), int(y * h)) for x, y in right_eye])
                
                # for point in left_eye + right_eye:
                #     cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)  # Draw eye landmarks in red
                
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                if ear < EAR_THRESHOLD:
                    frame_count += 1
                    if frame_count >= CONSECUTIVE_FRAMES:
                        drowsy = True
                else:
                    frame_count = 0
        
        if drowsy:
            if drowsy_status != "Drowsiness Detected!":  # Avoid sending duplicate commands
                drowsy_status = "Drowsiness Detected!"
                print("[DEBUG] Drowsiness Detected! Sending command to Arduino...")
                send_to_arduino("d")  # Send alert to Arduino
        else:
            drowsy_status = "Monitoring..."
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/drowsiness_status')
def drowsiness_status():
    return jsonify({"message": drowsy_status})

if __name__ == '__main__':
    app.run(debug=True)
