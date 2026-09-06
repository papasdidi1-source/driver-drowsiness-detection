import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import math
from pygame import mixer


# ==========================================
# INITIALIZATION
# ==========================================

model = load_model("models/model.h5")

mixer.init()
sound = mixer.Sound("alarm.wav")


# Haar Cascades
face = cv2.CascadeClassifier(
    "haar cascade files/haarcascade_frontalface_alt.xml"
)

leye = cv2.CascadeClassifier(
    "haar cascade files/haarcascade_lefteye_2splits.xml"
)

reye = cv2.CascadeClassifier(
    "haar cascade files/haarcascade_righteye_2splits.xml"
)


# MediaPipe
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# VARIABLES
# ==========================================

score = 0
max_score = 30

yawn_count = 0
mouth_open_frames = 0

alarm_on = False


# ==========================================
# DISTANCE FUNCTION
# ==========================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ==========================================
# PROCESS CAMERA FRAME
# ==========================================

def process_frame(frame):

    global score
    global yawn_count
    global mouth_open_frames
    global alarm_on


    height, width = frame.shape[:2]


    # --------------------------------------
    # GRAYSCALE
    # --------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------
    # FACE + EYE DETECTION
    # --------------------------------------

    faces = face.detectMultiScale(
        gray,
        minNeighbors=5,
        scaleFactor=1.1,
        minSize=(25, 25)
    )

    left_eye = leye.detectMultiScale(gray)

    right_eye = reye.detectMultiScale(gray)


    val1 = 1
    val2 = 1


    # --------------------------------------
    # FACE
    # --------------------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (100, 100, 100),
            1
        )


    # --------------------------------------
    # RIGHT EYE
    # --------------------------------------

    for (x, y, w, h) in right_eye:

        eye = frame[
            y:y+h,
            x:x+w
        ]

        eye = cv2.cvtColor(
            eye,
            cv2.COLOR_BGR2GRAY
        )

        eye = cv2.resize(
            eye,
            (52, 52)
        )

        eye = eye / 255.0

        eye = eye.reshape(
            52,
            52,
            1
        )

        prediction = model.predict(
            np.expand_dims(
                eye,
                axis=0
            ),
            verbose=0
        )

        if prediction[0][0] > prediction[0][1]:

            val1 = 0

        else:

            val1 = 1

        break


    # --------------------------------------
    # LEFT EYE
    # --------------------------------------

    for (x, y, w, h) in left_eye:

        eye = frame[
            y:y+h,
            x:x+w
        ]

        eye = cv2.cvtColor(
            eye,
            cv2.COLOR_BGR2GRAY
        )

        eye = cv2.resize(
            eye,
            (52, 52)
        )

        eye = eye / 255.0

        eye = eye.reshape(
            52,
            52,
            1
        )

        prediction = model.predict(
            np.expand_dims(
                eye,
                axis=0
            ),
            verbose=0
        )

        if prediction[0][0] > prediction[0][1]:

            val2 = 0

        else:

            val2 = 1

        break


    # --------------------------------------
    # EYE DROWSINESS
    # --------------------------------------

    if val1 == 0 and val2 == 0:

        score += 1

    else:

        score -= 2


    score = max(
        0,
        min(score, max_score)
    )


    eye_score = int(
        (score / max_score) * 100
    )


    # --------------------------------------
    # YAWN DETECTION
    # --------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    mouth_status = "CLOSED"


    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0]

        upper_lip = landmarks.landmark[13]

        lower_lip = landmarks.landmark[14]

        left_mouth = landmarks.landmark[61]

        right_mouth = landmarks.landmark[291]


        vertical = distance(
            upper_lip,
            lower_lip
        )

        horizontal = distance(
            left_mouth,
            right_mouth
        )


        if horizontal != 0:

            mar = vertical / horizontal

        else:

            mar = 0


        if mar > 0.35:

            mouth_open_frames += 1

            mouth_status = "OPEN"

        else:

            mouth_open_frames = 0

            mouth_status = "CLOSED"


        if mouth_open_frames >= 15:

            yawn_count += 1

            mouth_open_frames = 0


    # --------------------------------------
    # FINAL SCORE
    # --------------------------------------

    yawn_score = min(
        yawn_count * 5,
        20
    )


    final_score = min(
        eye_score + yawn_score,
        100
    )


    # --------------------------------------
    # STATUS
    # --------------------------------------

    if final_score < 30:

        status = "ALERT"

    elif final_score < 70:

        status = "WARNING"

    else:

        status = "DROWSY"


    # --------------------------------------
    # ALARM
    # --------------------------------------

    if final_score >= 70:

        if not alarm_on:

            sound.play(-1)

            alarm_on = True

    else:

        if alarm_on:

            sound.stop()

            alarm_on = False


    # --------------------------------------
    # DISPLAY
    # --------------------------------------

    cv2.putText(
        frame,
        f"Drowsiness: {final_score}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Yawns: {yawn_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Mouth: {mouth_status}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    if final_score >= 70:

        cv2.rectangle(
            frame,
            (0, 0),
            (width - 1, height - 1),
            (0, 0, 255),
            5
        )


    return frame, final_score, yawn_count, status, mouth_status


# ==========================================
# STOP ALARM
# ==========================================

def stop_alarm():

    global alarm_on

    if alarm_on:

        sound.stop()

        alarm_on = False