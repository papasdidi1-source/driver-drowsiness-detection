import cv2
import mediapipe as mp
import math

# -----------------------------
# MediaPipe Face Mesh
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

yawn_count = 0
mouth_open_frames = 0

# -----------------------------
# Function to calculate distance
# -----------------------------

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

# -----------------------------
# Main loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to access camera")
        break

    # Flip camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb_frame)

    mouth_status = "MOUTH CLOSED"

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # Mouth landmarks
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]

        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]

        # Calculate mouth opening
        vertical_distance = distance(
            upper_lip,
            lower_lip
        )

        horizontal_distance = distance(
            left_mouth,
            right_mouth
        )

        # Mouth Aspect Ratio
        if horizontal_distance != 0:

            mar = (
                vertical_distance /
                horizontal_distance
            )

        else:
            mar = 0

        # -----------------------------
        # Yawn detection
        # -----------------------------

        if mar > 0.35:

            mouth_open_frames += 1
            mouth_status = "MOUTH OPEN"

        else:

            mouth_open_frames = 0
            mouth_status = "MOUTH CLOSED"

        # Detect yawn after mouth stays open
        if mouth_open_frames >= 15:

            yawn_count += 1

            mouth_open_frames = 0

            cv2.putText(
                frame,
                "YAWN DETECTED!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    # -----------------------------
    # Display information
    # -----------------------------

    cv2.putText(
        frame,
        "Yawn Count: " + str(yawn_count),
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Status: " + mouth_status,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Yawn Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()
face_mesh.close()