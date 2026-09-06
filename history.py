import sqlite3
from datetime import datetime

DATABASE = "detection_history.db"


def create_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            time TEXT,
            drowsiness INTEGER,
            yawns INTEGER,
            status TEXT,
            alarm TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_detection(drowsiness, yawns, status, alarm):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    now = datetime.now()

    cursor.execute("""
        INSERT INTO detections
        (date, time, drowsiness, yawns, status, alarm)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        drowsiness,
        yawns,
        status,
        alarm
    ))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT date, time, drowsiness, yawns, status, alarm
        FROM detections
        ORDER BY id DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data


if __name__ == "__main__":

    create_database()

    print("Database created successfully!")

    save_detection(
        75,
        2,
        "DROWSY",
        "YES"
    )

    print("Test record saved!")

    history = get_history()

    for row in history:
        print(row)