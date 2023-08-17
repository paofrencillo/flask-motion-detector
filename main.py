from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import os
import requests


# Initialize Flask app
app = Flask(__name__)
socket = SocketIO(app, cors_allowed_origins="*")

def frames():
    # Setup Camera
    cap = cv2.VideoCapture(cv2.CAP_V4L2)
    # Assigning our static_back to None
    static_back = None

    while (cap.isOpened()):
        # Reading frame(image) from video     
        ret, frame = cap.read()
        if frame is None:
            exit()

        # Initializing motion = 0(no motion)
        motion = 0

        # Converting color image to gray_scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Converting gray scale image to GaussianBlur
        # so that change can be find easily
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # In first iteration we assign the value
        # of static_back to our first frame
        if static_back is None:
            static_back = gray
            continue

        # Difference between static background
        # and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(static_back, gray)

        # If change in between static background and
        # current frame is greater than 110 it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, 120, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Finding contour of moving object
        contours, _ = cv2.findContours(
            thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 10000:
                continue

            motion = 1

            if motion == 1:
                socket.emit("detect", {"isDetected": True})
            else:
                socket.emit("detect", {"isDetected": False})

            (x, y, w, h) = cv2.boundingRect(contour)
            # making red rectangle around the moving object
            # putting text above the rectangle
            box = cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 205, 50), 2)
            cv2.putText(
                box,
                "Movement Detected!",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 205, 50),
                1,
                cv2.LINE_AA,
            )

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/live")
def live():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    socket.run(app, debug=False, port=os.getenv("PORT", default=5000))
