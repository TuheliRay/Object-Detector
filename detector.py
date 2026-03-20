import cv2
import pyttsx3
from ultralytics import YOLO

# --- Setup ---
# Load the YOLO26 nano model (downloads automatically on first run)
model = YOLO("yolo26n.pt")

# Initialize the TTS engine once (creating it repeatedly causes lag)
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)  # Speaking speed (words per minute)

def speak(text):
    """Convert a string to speech and block until it's done speaking."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def detect_and_speak(frame):
    """
    Takes an image frame (numpy array from OpenCV),
    runs YOLO detection, draws boxes, and speaks detected objects.
    Returns the annotated frame.
    """
    results = model(frame, verbose=False)  # verbose=False silences console spam

    # results[0].boxes contains all detections for this frame
    detected_labels = []
    for box in results[0].boxes:
        # Each box has a class ID — we map it to a human-readable name
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])

        # Only announce objects we're fairly confident about
        if confidence > 0.8:
            detected_labels.append(label)

    # Build the annotated frame (YOLO draws boxes + labels for us)
    annotated_frame = results[0].plot()

    # Speak unique detected objects (use a set to avoid repeating "person person person")
    unique_labels = list(set(detected_labels))
    if unique_labels:
        speech_text = "I see: " + ", ".join(unique_labels)
        print(speech_text)
        speak(speech_text)
    else:
        speak("No objects detected")

    return annotated_frame