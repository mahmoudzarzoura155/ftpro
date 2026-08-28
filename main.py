import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import time
import csv
from pathlib import Path
from collections import Counter

from PIL import Image, ImageTk
from fer import FER


class EmotionDetectionGUI:

    # Initialize
    def __init__(self, root):

        self.root = root
        self.root.title("AI Emotion Detection System")
        self.root.geometry("1350x820")
        self.root.minsize(1100, 700)

        # Camera
        self.camera = None
        self.running = False

        # AI Model
        try:
            self.detector = FER(mtcnn=True)
        except Exception as e:
            self.detector = None
            messagebox.showerror(
                "Model Error", f"Could not load the emotion model:\n\n{e}"
            )

        # Performance
        self.start_time = None
        self.frame_count = 0
        self.fps = 0
        self.last_process_time = 0
        self.process_interval = 0.20
        self.last_faces = []

        # Statistics
        self.emotion_counter = Counter()

        # GUI Variables
        self.camera_status = tk.StringVar(value="Camera: OFF")
        self.face_count = tk.StringVar(value="Faces: 0")
        self.fps_text = tk.StringVar(value="FPS: 0")
        self.session_time = tk.StringVar(value="Session: 00:00")
        self.dominant_emotion = tk.StringVar(value="Dominant: ---")
        self.system_status = tk.StringVar(value="System: Ready")

        # Log File
        self.project_dir = Path(__file__).parent
        self.log_dir = self.project_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "emotion_log.csv"

        # Build GUI
        self.create_gui()

        self.root.bind("<q>", lambda event: self.stop_camera())
        self.root.bind("<Q>", lambda event: self.stop_camera())
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

    # Build GUI
    def create_gui(self):

        # Title
        title = ttk.Label(
            self.root, text="AI Emotion Detection System", font=("Segoe UI", 23, "bold")
        )
        title.pack(pady=(15, 2))

        subtitle = ttk.Label(
            self.root, text="Real-Time Facial Emotion Detection", font=("Segoe UI", 11)
        )
        subtitle.pack(pady=(0, 10))

        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # Camera Area
        camera_frame = ttk.LabelFrame(main_frame, text="Live Camera", padding=10)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = ttk.Label(
            camera_frame,
            text="Camera is OFF\n\nPress START CAMERA",
            anchor="center",
            font=("Segoe UI", 16),
        )
        self.video_label.pack(fill="both", expand=True)

        # Right Panel
        right_frame = ttk.Frame(main_frame, width=390)
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        # Camera Controls
        control_frame = ttk.LabelFrame(right_frame, text="Camera Controls", padding=10)
        control_frame.pack(fill="x", pady=(0, 10))

        self.start_button = ttk.Button(
            control_frame, text="▶ Start Camera", command=self.start_camera
        )
        self.start_button.pack(fill="x", pady=5)

        self.stop_button = ttk.Button(
            control_frame,
            text="■ Stop Camera",
            command=self.stop_camera,
            state="disabled",
        )
        self.stop_button.pack(fill="x", pady=5)

        # Live Status
        status_frame = ttk.LabelFrame(right_frame, text="Live Status", padding=10)
        status_frame.pack(fill="x", pady=(0, 10))

        self.create_stat(status_frame, "System", self.system_status)

        self.create_stat(status_frame, "Camera", self.camera_status)

        self.create_stat(status_frame, "Faces", self.face_count)

        self.create_stat(status_frame, "FPS", self.fps_text)

        self.create_stat(status_frame, "Dominant Emotion", self.dominant_emotion)

        self.create_stat(status_frame, "Session", self.session_time)

        # Emotion Statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Emotion Statistics", padding=10)
        stats_frame.pack(fill="x", pady=(0, 10))

        self.stats_text = tk.Text(
            stats_frame, height=10, state="disabled", font=("Consolas", 10)
        )
        self.stats_text.pack(fill="x")

        # Bottom Buttons
        bottom_frame = ttk.Frame(right_frame)
        bottom_frame.pack(fill="x")

        ttk.Button(
            bottom_frame, text="Clear Statistics", command=self.clear_statistics
        ).pack(side="left", expand=True, fill="x", padx=(0, 3))

        ttk.Button(bottom_frame, text="Save Log", command=self.save_log).pack(
            side="left", expand=True, fill="x", padx=3
        )

        ttk.Button(bottom_frame, text="Exit", command=self.close_application).pack(
            side="left", expand=True, fill="x", padx=(3, 0)
        )

    # Status Row
    def create_stat(self, parent, name, variable):

        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)

        ttk.Label(frame, text=f"{name}:", font=("Segoe UI", 10, "bold")).pack(
            side="left"
        )

        ttk.Label(frame, textvariable=variable).pack(side="right")

    # Start Camera
    def start_camera(self):

        if self.running:
            return

        if self.detector is None:
            messagebox.showerror("Model Error", "Emotion model is not available.")
            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            self.camera.release()
            self.camera = None

            messagebox.showerror("Camera Error", "Could not open the default camera.")

            return

        # Camera Size
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)

        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        # Reset Session
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.last_process_time = 0
        self.last_faces = []

        self.emotion_counter.clear()
        self.update_statistics()

        self.start_button.config(state="disabled")

        self.stop_button.config(state="normal")

        self.camera_status.set("Camera: ON")

        self.system_status.set("System: Running")

        self.update_camera()

    # Stop Camera
    def stop_camera(self):

        if not self.running:
            return

        self.running = False

        if self.camera:
            self.camera.release()
            self.camera = None

        self.start_button.config(state="normal")

        self.stop_button.config(state="disabled")

        self.camera_status.set("Camera: OFF")

        self.face_count.set("Faces: 0")

        self.dominant_emotion.set("Dominant: ---")

        self.system_status.set("System: Ready")

        self.video_label.configure(image="", text="Camera is OFF\n\nPress START CAMERA")

        self.video_label.image = None

    # Camera Loop
    def update_camera(self):

        if not self.running or self.camera is None:
            return

        ok, frame = self.camera.read()

        if not ok:

            self.system_status.set("System: Camera read failed")

            self.stop_camera()
            return

        # Mirror Frame
        frame = cv2.flip(frame, 1)

        current_time = time.time()

        # Emotion Detection
        if current_time - self.last_process_time >= self.process_interval:

            self.last_process_time = current_time

            try:

                self.last_faces = self.detector.detect_emotions(frame)

                self.process_faces(self.last_faces)

            except Exception:

                self.system_status.set("System: Detection Error")

        # Draw Results
        self.draw_faces(frame, self.last_faces)

        # FPS
        self.frame_count += 1

        elapsed = current_time - self.start_time

        if elapsed > 0:
            self.fps = self.frame_count / elapsed

        self.fps_text.set(f"FPS: {self.fps:.1f}")

        # Session Time
        minutes = int(elapsed // 60)

        seconds = int(elapsed % 60)

        self.session_time.set(f"Session: {minutes:02d}:{seconds:02d}")

        # Convert Frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame_rgb)

        image.thumbnail((850, 620), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=photo, text="")

        self.video_label.image = photo

        # Next Frame
        self.root.after(10, self.update_camera)

    # Process Emotions
    def process_faces(self, faces):

        self.face_count.set(f"Faces: {len(faces)}")

        if not faces:

            self.dominant_emotion.set("Dominant: ---")

            return

        current_emotions = []

        # Process Faces
        for face in faces:

            emotions = face.get("emotions", {})

            if not emotions:
                continue

            emotion = max(emotions, key=emotions.get)

            confidence = emotions[emotion]

            current_emotions.append(emotion)

            self.emotion_counter[emotion] += 1

            # Save Detection
            if confidence >= 0.40:

                x, y, w, h = face["box"]

                self.write_csv(emotion, confidence, x, y, w, h)

        # Dominant Emotion
        if current_emotions:

            dominant = Counter(current_emotions).most_common(1)[0][0]

            self.dominant_emotion.set(f"Dominant: {dominant}")

        self.update_statistics()

    # Draw Results
    def draw_faces(self, frame, faces):

        for index, face in enumerate(faces):

            x, y, w, h = face["box"]

            emotions = face.get("emotions", {})

            if not emotions:
                continue

            emotion = max(emotions, key=emotions.get)

            confidence = emotions[emotion]

            # Face Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

            # Emotion Label
            label = f"Face {index + 1}: " f"{emotion} " f"{confidence:.0%}"

            label_y = max(25, y - 10)

            cv2.putText(
                frame,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 0),
                2,
            )

        # Global Info
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    # Save CSV
    def write_csv(self, emotion, confidence, x, y, w, h):

        new_file = not self.log_file.exists()

        try:

            with open(self.log_file, "a", newline="", encoding="utf-8") as file:

                writer = csv.writer(file)

                if new_file:

                    writer.writerow(
                        ["timestamp", "emotion", "confidence", "x", "y", "w", "h"]
                    )

                writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        emotion,
                        round(confidence, 3),
                        x,
                        y,
                        w,
                        h,
                    ]
                )

        except Exception:

            self.system_status.set("System: CSV Error")

    # Update Statistics
    def update_statistics(self):

        self.stats_text.config(state="normal")

        self.stats_text.delete("1.0", "end")

        self.stats_text.insert("end", "EMOTIONS\n")

        self.stats_text.insert("end", "-------------------------\n")

        if self.emotion_counter:

            total = sum(self.emotion_counter.values())

            for emotion, count in self.emotion_counter.most_common():

                percentage = count / total * 100

                self.stats_text.insert(
                    "end", f"{emotion:<12}" f"{count:>5} " f"({percentage:5.1f}%)\n"
                )

            self.stats_text.insert("end", f"\nTotal detections: {total}\n")

        else:

            self.stats_text.insert("end", "No emotion data yet.\n")

        self.stats_text.config(state="disabled")

    # Clear Statistics
    def clear_statistics(self):

        self.emotion_counter.clear()

        self.dominant_emotion.set("Dominant: ---")

        self.update_statistics()

        self.system_status.set("System: Statistics cleared")

    # Save Log
    def save_log(self):

        try:
            self.log_dir.mkdir(exist_ok=True)

            if not self.log_file.exists():
                with open(self.log_file, "w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ["timestamp", "emotion", "confidence", "x", "y", "w", "h"]
                    )

            messagebox.showinfo(
                "Saved",
                f"Emotion log saved successfully.\n\nLocation:\n{self.log_file}",
            )
            self.system_status.set("System: Log saved")

        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # Close App
    def close_application(self):

        if self.running:
            self.stop_camera()

        self.root.destroy()


# Run App
if __name__ == "__main__":

    root = tk.Tk()

    app = EmotionDetectionGUI(root)

    root.mainloop()
