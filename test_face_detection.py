"""
Quick smoke test for Test Case #1 in the Testing Plan:
"Face detected correctly — run with 3 different people, varied lighting."

This script isolates JUST face detection (no emotion model, no FER/TensorFlow
dependency) so you can confirm the camera and OpenCV pipeline work before
troubleshooting the heavier emotion-classification stage in main.py.

Usage:
    python test_face_detection.py [--camera 0]

Press 'q' to quit.
"""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Face detection smoke test")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Face detection smoke test running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

        status = f"Faces detected: {len(faces)}"
        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imshow("Face Detection Test - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
