import cv2
from fer import FER


def main():

    detector = FER(mtcnn=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    window = "Face Detection Test"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 800, 540)

    while True:

        ok, frame = camera.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)

        try:
            faces = detector.detect_emotions(frame)

        except Exception as e:
            print(f"Detection Error: {e}")
            break

        for index, face in enumerate(faces):

            x, y, w, h = face["box"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 220, 0), 
                2,
            )

            cv2.putText(
                frame,
                f"Face {index + 1}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
