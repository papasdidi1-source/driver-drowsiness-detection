import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TO_NUMBER = os.getenv("EMERGENCY_PHONE")


def send_emergency_sms(drowsiness, yawns):

    if not all([
        ACCOUNT_SID,
        AUTH_TOKEN,
        FROM_NUMBER,
        TO_NUMBER
    ]):
        print("Twilio settings are missing.")
        return False

    try:

        client = Client(
            ACCOUNT_SID,
            AUTH_TOKEN
        )

        message = client.messages.create(
            body=(
                "🚨 DRIVER DROWSINESS ALERT!\n"
                f"Drowsiness: {drowsiness}%\n"
                f"Yawns: {yawns}\n"
                "Please check the driver."
            ),
            from_=FROM_NUMBER,
            to=TO_NUMBER
        )

        print(
            "Emergency SMS sent:",
            message.sid
        )

        return True

    except Exception as e:

        print(
            "SMS error:",
            e
        )

        return False