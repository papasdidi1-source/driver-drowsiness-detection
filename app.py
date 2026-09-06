import streamlit as st
import pandas as pd
import sqlite3
import threading
import time
import av

from streamlit_webrtc import webrtc_streamer
from camera import process_frame, stop_alarm
from history import save_detection


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Driver Safety AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
    margin-bottom: 15px;
}

.metric-title {
    font-size: 16px;
    font-weight: 500;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SHARED CAMERA DATA
# ============================================================

lock = threading.Lock()

camera_data = {
    "drowsiness": 0,
    "yawns": 0,
    "status": "ALERT",
    "mouth": "CLOSED"
}

last_saved_time = 0


# ============================================================
# DATABASE
# ============================================================

def get_history():

    conn = sqlite3.connect(
        "detection_history.db"
    )

    query = """
        SELECT
            date,
            time,
            drowsiness,
            yawns,
            status,
            alarm
        FROM detections
        ORDER BY id DESC
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df


# ============================================================
# CAMERA CALLBACK
# ============================================================

def video_frame_callback(frame):

    global last_saved_time

    img = frame.to_ndarray(
        format="bgr24"
    )

    processed_frame, drowsiness, yawns, status, mouth = process_frame(
        img
    )

    with lock:

        camera_data["drowsiness"] = drowsiness
        camera_data["yawns"] = yawns
        camera_data["status"] = status
        camera_data["mouth"] = mouth

    current_time = time.time()

    if current_time - last_saved_time >= 5:

        alarm_status = (
            "YES"
            if drowsiness >= 70
            else "NO"
        )

        save_detection(
            drowsiness,
            yawns,
            status,
            alarm_status
        )

        last_saved_time = current_time

    return av.VideoFrame.from_ndarray(
        processed_frame,
        format="bgr24"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🚗 Driver Safety AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Driver Drowsiness & Yawn Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚗 Driver Safety")

st.sidebar.markdown("### System Features")

st.sidebar.success("✓ CNN Eye Detection")
st.sidebar.success("✓ Drowsiness Detection")
st.sidebar.success("✓ Yawn Detection")
st.sidebar.success("✓ Drowsiness Score")
st.sidebar.success("✓ Alarm System")
st.sidebar.success("✓ Detection History")
st.sidebar.success("✓ Live Camera")

st.sidebar.divider()

st.sidebar.info(
    "This system uses Computer Vision, "
    "CNN and MediaPipe to monitor driver alertness."
)


# ============================================================
# LIVE CAMERA
# ============================================================

st.markdown(
    '<div class="section-title">📷 Live Driver Monitoring</div>',
    unsafe_allow_html=True
)

st.info(
    "Click START below and allow camera access."
)

ctx = webrtc_streamer(
    key="driver-drowsiness-camera",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    media_toggle_controls=True
)


# ============================================================
# LIVE METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Live Detection</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with lock:

    drowsiness = camera_data["drowsiness"]
    yawns = camera_data["yawns"]
    status = camera_data["status"]
    mouth = camera_data["mouth"]


with col1:

    st.metric(
        "😴 Drowsiness",
        f"{drowsiness}%"
    )


with col2:

    st.metric(
        "🥱 Yawns",
        yawns
    )


with col3:

    st.metric(
        "🚦 Status",
        status
    )


with col4:

    st.metric(
        "👄 Mouth",
        mouth
    )


# ============================================================
# STATUS
# ============================================================

if status == "DROWSY":

    st.error(
        "🚨 HIGH DROWSINESS DETECTED — TAKE A BREAK!"
    )

elif status == "WARNING":

    st.warning(
        "⚠️ Signs of drowsiness detected."
    )

else:

    st.success(
        "✅ Driver is currently alert."
    )


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Driving Analysis</div>',
    unsafe_allow_html=True
)

df = get_history()


if not df.empty:

    chart_df = df.copy()

    chart_df = chart_df.sort_values(
        by=["date", "time"]
    )

    chart_df["Date-Time"] = (
        chart_df["date"]
        + " "
        + chart_df["time"]
    )

    chart_df = chart_df.set_index(
        "Date-Time"
    )


    # -----------------------------------------
    # DROWSINESS
    # -----------------------------------------

    st.write("### 😴 Drowsiness Trend")

    st.line_chart(
        chart_df["drowsiness"]
    )


    # -----------------------------------------
    # YAWNS
    # -----------------------------------------

    st.write("### 🥱 Yawn Activity")

    st.bar_chart(
        chart_df["yawns"]
    )


    # -----------------------------------------
    # STATISTICS
    # -----------------------------------------

    st.write("### 📊 Session Statistics")

    stat1, stat2, stat3 = st.columns(3)

    with stat1:

        st.metric(
            "Average Drowsiness",
            f"{df['drowsiness'].mean():.1f}%"
        )

    with stat2:

        st.metric(
            "Maximum Drowsiness",
            f"{df['drowsiness'].max()}%"
        )

    with stat3:

        st.metric(
            "Total Yawns",
            int(df['yawns'].max())
        )


    # -----------------------------------------
    # TABLE
    # -----------------------------------------

    st.write("### 📋 Recent Detection Records")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No detection history available yet. "
        "Start the camera to generate data."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🚗 Driver Safety AI | '
    'Computer Vision + CNN + MediaPipe'
    '</div>',
    unsafe_allow_html=True
)