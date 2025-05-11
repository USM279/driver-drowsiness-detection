import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import subprocess
import threading

# EAR calculation
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Load model the VGG16 for best results
model = load_model('vgg16_transfer_model.h5')
IMG_SIZE = (150, 150)

# Mediapipe setup which is a library From GOOGLE for AI to recognize a face and eyes ETC
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# EAR parameters ( Eye Aspect Ratio )
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 10
ear_counter = 0

# Model prediction history
model_history = deque(maxlen=15)

# Alarm state
alarm_playing = False
alarm_process = None  # store subprocess reference

# Function to play sound
def play_alarm():
    global alarm_playing, alarm_process
    alarm_process = subprocess.Popen(["afplay", "alert.wav"])
    alarm_process.wait()
    alarm_playing = False

# Start webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    is_drowsy_by_eye = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_idx = [362, 385, 387, 263, 373, 380]
            right_eye_idx = [33, 160, 158, 133, 153, 144]

            left_eye = np.array([(int(face_landmarks.landmark[i].x * w),
                                  int(face_landmarks.landmark[i].y * h)) for i in left_eye_idx])
            right_eye = np.array([(int(face_landmarks.landmark[i].x * w),
                                   int(face_landmarks.landmark[i].y * h)) for i in right_eye_idx])

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                ear_counter += 1
            else:
                ear_counter = 0

            if ear_counter >= EAR_CONSEC_FRAMES:
                is_drowsy_by_eye = True

    # Predict with model
    resized = cv2.resize(frame, IMG_SIZE)
    model_input = np.expand_dims(resized, axis=0) / 255.0
    pred = model.predict(model_input)
    is_drowsy_by_model = pred[0][0] > 0.5
    model_history.append(int(is_drowsy_by_model))

    drowsy_by_model = sum(model_history) > len(model_history) * 0.6

    # Final condition: both methods agree
    if is_drowsy_by_eye and drowsy_by_model:
        label = "DROWSY - WAKE UP"
        color = (0, 0, 255)
        if not alarm_playing:
            alarm_playing = True
            threading.Thread(target=play_alarm).start()
    else:
        label = "Alert"
        color = (0, 255, 0)
        if alarm_process and alarm_process.poll() is None:
            alarm_process.terminate()
            alarm_process = None
            alarm_playing = False

    # Show status on screen
    cv2.putText(frame, f"Status: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
