import cv2
from detector import detect_and_speak

cap = cv2.VideoCapture(0)  # 0 = default webcam

print("Press SPACE to detect objects in current frame. Press Q to quit.")

while True:
    ret, frame = cap.read()  # Grab the latest frame
    if not ret:
        print("Couldn't access webcam.")
        break
    # Show the live feed (unprocessed, so it's smooth)
    cv2.imshow("Live Feed - Press SPACE to detect", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Spacebar triggers detection
        print("Detecting...")
        annotated = detect_and_speak(frame)
        cv2.imshow("Detection Result", annotated)

    elif key == ord('q'):  # Q quits
        break

cap.release()
cv2.destroyAllWindows()