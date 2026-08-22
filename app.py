import gradio as gr
import cv2
import numpy as np
from fer import FER

# =========================================================
# Load FER Model
# =========================================================

detector = FER(mtcnn=False)


# =========================================================
# Emotion Detection
# =========================================================

def detect_emotion(image):
    if image is None:
        return None, "No image provided."

    # Gradio gives RGB image
    frame = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    try:
        faces = detector.detect_emotions(frame)
    except Exception as e:
        return None, f"Detection error: {e}"

    result_frame = frame.copy()

    results = []

    for index, face in enumerate(faces):

        x, y, w, h = face["box"]

        emotions = face.get(
            "emotions",
            {}
        )

        if not emotions:
            continue

        # Get highest emotion
        emotion = max(
            emotions,
            key=emotions.get
        )

        confidence = emotions[emotion]

        results.append(
            f"Face {index + 1}: "
            f"{emotion} "
            f"({confidence:.1%})"
        )

        # -------------------------------------------------
        # Face Rectangle
        # -------------------------------------------------

        cv2.rectangle(
            result_frame,
            (x, y),
            (x + w, y + h),
            (0, 220, 0),
            2
        )

        # -------------------------------------------------
        # Emotion Label
        # -------------------------------------------------

        label = (
            f"Face {index + 1}: "
            f"{emotion} "
            f"{confidence:.0%}"
        )

        label_y = max(
            25,
            y - 10
        )

        cv2.putText(
            result_frame,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2
        )

    # -----------------------------------------------------
    # Number of Faces
    # -----------------------------------------------------

    cv2.putText(
        result_frame,
        f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    # -----------------------------------------------------
    # Convert BGR -> RGB
    # -----------------------------------------------------

    result_frame = cv2.cvtColor(
        result_frame,
        cv2.COLOR_BGR2RGB
    )

    if results:

        result_text = "\n".join(results)

    else:

        result_text = (
            "No face detected.\n"
            "Please make sure your face "
            "is clearly visible."
        )

    return result_frame, result_text


# =========================================================
# Gradio Interface
# =========================================================

with gr.Blocks(
    title="AI Emotion Detection System"
) as demo:

    gr.Markdown(
        """
        # AI Emotion Detection System

        ### Real-Time Facial Emotion Detection

        This system uses computer vision and the FER
        emotion recognition model to detect facial emotions.
        """
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Camera / Image"
            )

            detect_button = gr.Button(
                "Detect Emotion",
                variant="primary"
            )

        with gr.Column():

            output_image = gr.Image(
                label="Detection Result"
            )

            output_text = gr.Textbox(
                label="Detection Results",
                lines=8
            )

    detect_button.click(
        fn=detect_emotion,
        inputs=input_image,
        outputs=[
            output_image,
            output_text
        ]
    )

    gr.Markdown(
        """
        ---
        **AI & Robotics Field Training — Emotion Detection Project**
        """
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    demo.launch()