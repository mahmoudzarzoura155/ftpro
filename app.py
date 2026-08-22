import streamlit as st
import cv2
import numpy as np
from fer import FER
from collections import Counter

st.set_page_config(
    page_title="AI Emotion Detection System",
    page_icon="😊",
    layout="wide"
)

# ---------------------------------------------------------
# Page Title
# ---------------------------------------------------------

st.title("AI Emotion Detection System")
st.caption("Real-Time Facial Emotion Detection")

# ---------------------------------------------------------
# Load FER Model
# ---------------------------------------------------------

@st.cache_resource
def load_detector():
    return FER(mtcnn=False)

detector = load_detector()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("System Controls")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.05
)

st.sidebar.info(
    "Take a photo using your webcam. "
    "The FER model will detect faces and predict emotions."
)

# ---------------------------------------------------------
# Camera Input
# ---------------------------------------------------------

picture = st.camera_input(
    "Take a picture using your camera"
)

# ---------------------------------------------------------
# Detection
# ---------------------------------------------------------

if picture is not None:

    bytes_data = picture.getvalue()

    np_array = np.frombuffer(
        bytes_data,
        np.uint8
    )

    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        st.error("Could not read the camera image.")
        st.stop()

    # Mirror image
    frame = cv2.flip(frame, 1)

    # -----------------------------------------------------
    # Emotion Detection
    # -----------------------------------------------------

    with st.spinner("Detecting emotions..."):

        try:
            faces = detector.detect_emotions(frame)

        except Exception as e:
            st.error(f"Detection error: {e}")
            st.stop()

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    emotions_detected = []

    for index, face in enumerate(faces):

        x, y, w, h = face["box"]

        emotions = face.get(
            "emotions",
            {}
        )

        if not emotions:
            continue

        emotion = max(
            emotions,
            key=emotions.get
        )

        confidence = emotions[emotion]

        emotions_detected.append(emotion)

        # -------------------------------------------------
        # Draw Face Rectangle
        # -------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 220, 0),
            2
        )

        # -------------------------------------------------
        # Draw Emotion
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
            frame,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2
        )

    # -----------------------------------------------------
    # Display Image
    # -----------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        frame_rgb,
        caption="Emotion Detection Result",
        use_container_width=True
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    st.subheader("Detection Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Faces Detected",
        len(faces)
    )

    if emotions_detected:

        dominant = Counter(
            emotions_detected
        ).most_common(1)[0][0]

        col2.metric(
            "Dominant Emotion",
            dominant
        )

        col3.metric(
            "Total Detections",
            len(emotions_detected)
        )

    else:

        col2.metric(
            "Dominant Emotion",
            "---"
        )

        col3.metric(
            "Total Detections",
            0
        )

    # -----------------------------------------------------
    # Detailed Emotion Results
    # -----------------------------------------------------

    if faces:

        st.subheader("Emotion Details")

        for index, face in enumerate(faces):

            emotions = face.get(
                "emotions",
                {}
            )

            if not emotions:
                continue

            emotion = max(
                emotions,
                key=emotions.get
            )

            confidence = emotions[emotion]

            st.write(
                f"**Face {index + 1}: "
                f"{emotion} "
                f"({confidence:.1%})**"
            )

            st.progress(
                float(confidence)
            )

            if confidence >= confidence_threshold:
                st.success(
                    f"High confidence: {emotion}"
                )
            else:
                st.warning(
                    f"Low confidence: {emotion}"
                )

    else:

        st.warning(
            "No face detected. "
            "Please make sure your face is clearly visible."
        )

else:

    st.info(
        "Click the camera button above "
        "to take a photo and detect emotions."
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "AI Emotion Detection System | "
    "Computer Vision & Artificial Intelligence"
)