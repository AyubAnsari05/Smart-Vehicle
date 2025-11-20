#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import subprocess  # to run drunk_detector.py
from tensorflow.keras.models import load_model
import hardware  # your custom hardware.py
import sys
import requests
import geocoder


MODEL_PATH = "pain_model.h5"   
WINDOW_SIZE = 30
DISPLAY_SCALE = 1.0
CONF_THRESH = 0.5


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



def load_trained_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] Model file '{path}' not found.")
        return None
    print("[INFO] Loading model from:", path)
    model = load_model(path, compile=False)
    print("[INFO] Model loaded.")
    return model



def extract_ratios_from_landmarks(face_landmarks, h, w):
    lm = face_landmarks.landmark

    left_eye  = np.array([lm[33].x * w, lm[33].y * h])
    right_eye = np.array([lm[263].x * w, lm[263].y * h])
    nose_tip  = np.array([lm[1].x * w, lm[1].y * h])
    mouth_top = np.array([lm[13].x * w, lm[13].y * h])
    mouth_bottom = np.array([lm[14].x * w, lm[14].y * h])
    chin = np.array([lm[152].x * w, lm[152].y * h])

    eye_distance = np.linalg.norm(left_eye - right_eye)
    nose_to_mouth = np.linalg.norm(nose_tip - mouth_top)
    face_height = np.linalg.norm(nose_tip - chin)
    mouth_open = np.linalg.norm(mouth_top - mouth_bottom)

    return np.array([
        eye_distance / face_height,
        nose_to_mouth / face_height,
        mouth_open / face_height
    ], dtype=np.float32)


BOT_TOKEN = " "
CHAT_ID = " "



def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data)
        print("[INFO] Telegram message sent.")
    except Exception as e:
        print("[ERROR] Could not send Telegram message:", e)

import geocoder

def get_location_link():
    try:
        g = geocoder.ip("me")
        if g.ok:
            lat, lng = g.latlng
            return f"https://www.google.com/maps?q={lat},{lng}"
        else:
            return "Location unavailable"
    except Exception as e:
        return "Location error"



def main():
    print("[INFO] Starting Pain Detector...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    model = load_trained_model(MODEL_PATH)
    if model is None:
        return

    buffer = []
    last_time = time.time()

    _notified = False

    label =""

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while True:
            # ✅ check hardware signal
            if hardware.check_signal():   
                print("[INFO] Hardware signal received → Switching to drunk_detector.py")
                cap.release()
                cv2.destroyAllWindows()
                subprocess.run([sys.executable, "drunk_detector.py"])
                return

            ret, frame = cap.read()
            if not ret:
                print("[WARN] Empty frame, stopping.")
                break

            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            pred_text = "Waiting..."

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ratios = extract_ratios_from_landmarks(face_landmarks, h, w)
                if ratios is not None:
                    buffer.append(ratios)
                    if len(buffer) > WINDOW_SIZE:
                        buffer.pop(0)

                # Draw mesh
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))

            # Make prediction once buffer is full
            if len(buffer) == WINDOW_SIZE:
                seq = np.mean(buffer, axis=0).reshape(1, -1)
                preds = model.predict(seq)
                p = float(preds.ravel()[0])
                prob_pain, prob_nopain = p, 1.0 - p
                label = "PAIN" if prob_pain >= CONF_THRESH else "NO PAIN"
                pred_text = f"{label} | Pain: {prob_pain:.2f}  NoPain: {prob_nopain:.2f}"

            if label == "PAIN" and not _notified:
                location_url = get_location_link()
                send_telegram_message(f"⚠️ Pain detected! Please check immediately.\n📍 Location: {location_url}")
                _notified = True

            # FPS counter
            cur_time = time.time()
            fps = 1.0 / (cur_time - last_time + 1e-8)
            last_time = cur_time

            # Display
            cv2.putText(image, pred_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            disp = cv2.resize(image, (0,0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("PainDetector", disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
