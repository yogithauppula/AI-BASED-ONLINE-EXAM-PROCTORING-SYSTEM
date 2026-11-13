import cv2
import mediapipe as mp
import time
import sounddevice as sd
import numpy as np
import threading

# ------------------ SELECT MICROPHONE ------------------
print(sd.query_devices())  # 👈 check your mic index from here
MIC_INDEX = 0  # ⚠️ CHANGE THIS based on output (important)

# ------------------ AUDIO GLOBAL ------------------
audio_flag = False

# ------------------ AUDIO THREAD FUNCTION ------------------
def audio_monitor():
    global audio_flag
    duration = 1
    sample_rate = 44100

    print("Audio thread started...")

    while True:
        try:
            audio = sd.rec(int(duration * sample_rate),
                           samplerate=sample_rate,
                           channels=1,
                           device=MIC_INDEX)
            sd.wait()

            volume = np.linalg.norm(audio)
            print("Volume:", volume)  # DEBUG

            if volume > 0.02:   # 🔥 tuned for your system
                audio_flag = True
                print("Noise Detected!")
            else:
                audio_flag = False

        except Exception as e:
            print("Audio Error:", e)


# ------------------ START AUDIO THREAD ------------------
audio_thread = threading.Thread(target=audio_monitor, daemon=True)
audio_thread.start()

# ------------------ INITIAL SETUP ------------------
cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

log_file = open("log.txt", "a")

# Cooldown
last_capture_time = 0
cooldown = 5

# Calibration
center_x = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_time = time.time()

    # ------------------ FACE DETECTION ------------------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        cv2.putText(frame, "No Face", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif len(faces) > 1:
        cv2.putText(frame, "Multiple Faces", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if current_time - last_capture_time > cooldown:
            cv2.imwrite(f"multiple_{current_time}.jpg", frame)
            log_file.write("Multiple faces detected\n")
            last_capture_time = current_time

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # ------------------ HEAD TRACKING ------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            nose = face_landmarks.landmark[1]

            h, w, _ = frame.shape
            x = int(nose.x * w)
            y = int(nose.y * h)

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # ----------- CALIBRATION -----------
            if center_x is None:
                center_x = nose.x

            diff = nose.x - center_x

            # ----------- DETECTION -----------
            if diff < -0.05:
                cv2.putText(frame, "Looking LEFT", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if current_time - last_capture_time > cooldown:
                    cv2.imwrite(f"left_{current_time}.jpg", frame)
                    log_file.write("Looking LEFT\n")
                    last_capture_time = current_time

            elif diff > 0.05:
                cv2.putText(frame, "Looking RIGHT", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if current_time - last_capture_time > cooldown:
                    cv2.imwrite(f"right_{current_time}.jpg", frame)
                    log_file.write("Looking RIGHT\n")
                    last_capture_time = current_time

            elif nose.y > 0.65:
                cv2.putText(frame, "Looking DOWN", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if current_time - last_capture_time > cooldown:
                    cv2.imwrite(f"down_{current_time}.jpg", frame)
                    log_file.write("Looking DOWN\n")
                    last_capture_time = current_time

    # ------------------ AUDIO DETECTION ------------------
    if audio_flag:
        cv2.putText(frame, "Noise Detected", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if current_time - last_capture_time > cooldown:
            cv2.imwrite(f"noise_{current_time}.jpg", frame)
            log_file.write("Noise detected\n")
            last_capture_time = current_time

    # ------------------ STATUS ------------------
    cv2.putText(frame, "Monitoring...", (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, "Press R to recalibrate", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("AI Proctoring System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        center_x = None

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()