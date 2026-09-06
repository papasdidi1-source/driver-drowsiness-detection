import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import mediapipe as mp
import math
import time

from history import create_database, save_detection


# ==========================================
# ALARM
# ==========================================

mixer.init()
sound = mixer.Sound('alarm.wav')


# ==========================================
# HAAR CASCADES
# ==========================================

face = cv2.CascadeClassifier(
    'haar cascade files/haarcascade_frontalface_alt.xml'
)

leye = cv2.CascadeClassifier(
    'haar cascade files/haarcascade_lefteye_2splits.xml'
)

reye = cv2.CascadeClassifier(
    'haar cascade files/haarcascade_righteye_2splits.xml'
)


# ==========================================
# CNN MODEL
# ==========================================

model = load_model('models/model.h5')


# ==========================================
# MEDIAPIPE FACE MESH
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# DATABASE
# ==========================================

create_database()


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


# ==========================================
# VARIABLES
# ==========================================

score = 0
max_score = 30

yawn_count = 0
mouth_open_frames = 0

thicc = 2
alarm_on = False

# Save database record every 5 seconds
last_saved_time = time.time()


# ==========================================
# DISTANCE FUNCTION
# ==========================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Unable to access camera")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]


    # ======================================
    # GRAYSCALE
    # ======================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ======================================
    # FACE + EYE DETECTION
    # ======================================

    faces = face.detectMultiScale(
        gray,
        minNeighbors=5,
        scaleFactor=1.1,
        minSize=(25, 25)
    )

    left_eye = leye.detectMultiScale(gray)

    right_eye = reye.detectMultiScale(gray)


    # Reset eye status
    val1 = 1
    val2 = 1


    # ======================================
    # DRAW FACE
    # ======================================

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (100, 100, 100),
            1
        )


    # ======================================
    # RIGHT EYE
    # ======================================

    for (x, y, w, h) in right_eye:

        r_eye = frame[
            y:y+h,
            x:x+w
        ]

        r_eye = cv2.cvtColor(
            r_eye,
            cv2.COLOR_BGR2GRAY
        )

        r_eye = cv2.resize(
            r_eye,
            (52, 52)
        )

        r_eye = r_eye / 255

        r_eye = r_eye.reshape(
            52,
            52,
            1
        )

        rpred = model.predict(
            np.expand_dims(
                r_eye,
                axis=0
            ),
            verbose=0
        )

        if rpred[0][0] > rpred[0][1]:

            val1 = 0

        else:

            val1 = 1

        break


    # ======================================
    # LEFT EYE
    # ======================================

    for (x, y, w, h) in left_eye:

        l_eye = frame[
            y:y+h,
            x:x+w
        ]

        l_eye = cv2.cvtColor(
            l_eye,
            cv2.COLOR_BGR2GRAY
        )

        l_eye = cv2.resize(
            l_eye,
            (52, 52)
        )

        l_eye = l_eye / 255

        l_eye = l_eye.reshape(
            52,
            52,
            1
        )

        lpred = model.predict(
            np.expand_dims(
                l_eye,
                axis=0
            ),
            verbose=0
        )

        if lpred[0][0] > lpred[0][1]:

            val2 = 0

        else:

            val2 = 1

        break


    # ======================================
    # EYE-BASED DROWSINESS
    # ======================================

    if val1 == 0 and val2 == 0:

        score += 1

    else:

        score -= 2


    # Keep score within range

    if score < 0:

        score = 0

    if score > max_score:

        score = max_score


    # ======================================
    # MEDIAPIPE FACE MESH
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(
        rgb_frame
    )


    mouth_status = "CLOSED"


    # ======================================
    # YAWN DETECTION
    # ======================================

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0]


        # Upper and lower lip
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]


        # Left and right side of mouth
        left_mouth = landmarks.landmark[61]
        right_mouth = landmarks.landmark[291]


        # Calculate distances

        vertical_distance = distance(
            upper_lip,
            lower_lip
        )

        horizontal_distance = distance(
            left_mouth,
            right_mouth
        )


        # Calculate Mouth Aspect Ratio

        if horizontal_distance != 0:

            mar = (
                vertical_distance /
                horizontal_distance
            )

        else:

            mar = 0


        # Check if mouth is open

        if mar > 0.35:

            mouth_open_frames += 1

            mouth_status = "OPEN"

        else:

            mouth_open_frames = 0

            mouth_status = "CLOSED"


        # Detect yawn

        if mouth_open_frames >= 15:

            yawn_count += 1

            mouth_open_frames = 0


    # ======================================
    # DROWSINESS PERCENTAGE
    # ======================================

    drowsiness_percentage = int(
        (score / max_score) * 100
    )


    # ======================================
    # YAWN CONTRIBUTION
    # ======================================

    yawn_score = min(
        yawn_count * 5,
        20
    )


    # Final drowsiness score

    final_score = min(
        drowsiness_percentage + yawn_score,
        100
    )


    # ======================================
    # STATUS
    # ======================================

    if final_score < 30:

        status = "ALERT"

    elif final_score < 70:

        status = "WARNING"

    else:

        status = "DROWSY"


    # ======================================
    # SAVE DETECTION HISTORY
    # ======================================

    current_time = time.time()


    if current_time - last_saved_time >= 5:

        if final_score >= 70:

            alarm_status = "YES"

        else:

            alarm_status = "NO"


        save_detection(
            final_score,
            yawn_count,
            status,
            alarm_status
        )


        last_saved_time = current_time


    # ======================================
    # DISPLAY PANEL
    # ======================================

    cv2.rectangle(
        frame,
        (0, height - 155),
        (400, height),
        (0, 0, 0),
        thickness=cv2.FILLED
    )


    # Drowsiness

    cv2.putText(
        frame,
        "Drowsiness: "
        + str(final_score)
        + "%",
        (10, height - 115),
        font,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # Status

    cv2.putText(
        frame,
        "Status: "
        + status,
        (10, height - 80),
        font,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # Yawn count

    cv2.putText(
        frame,
        "Yawns: "
        + str(yawn_count),
        (10, height - 45),
        font,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # Mouth status

    cv2.putText(
        frame,
        "Mouth: "
        + mouth_status,
        (190, height - 45),
        font,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # ======================================
    # ALARM
    # ======================================

    if final_score >= 70:

        if not alarm_on:

            try:

                sound.play(-1)

                alarm_on = True

            except Exception as e:

                print(
                    "Alarm error:",
                    e
                )


        # Red border

        if thicc < 16:

            thicc += 2

        else:

            thicc -= 2

            if thicc < 2:

                thicc = 2


        cv2.rectangle(
            frame,
            (0, 0),
            (width, height),
            (0, 0, 255),
            thicc
        )


        # Warning text

        cv2.putText(
            frame,
            "!!! DROWSINESS ALERT !!!",
            (
                width // 2 - 180,
                50
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    else:

        if alarm_on:

            sound.stop()

            alarm_on = False

        thicc = 2


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )


    # ======================================
    # PRESS Q TO QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


# ==========================================
# CLEANUP
# ==========================================

sound.stop()

cap.release()

cv2.destroyAllWindows()

face_mesh.close()