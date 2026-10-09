
import cv2
import threading
import queue
from ultralytics import YOLO

# ==========================================
# HUMAN IN-OUT COUNTER WITH VOICE
# ==========================================

WIDTH = 900
HEIGHT = 600
LINE_X = WIDTH // 2
OFFSET = 15

# ---------------- VOICE ----------------

voice_queue = queue.Queue()

def voice_worker():
    try:
        import pyttsx3

        engine = pyttsx3.init("sapi5")
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)

        while True:
            message = voice_queue.get()

            if message is None:
                break

            engine.say(message)
            engine.runAndWait()
            voice_queue.task_done()

        engine.stop()

    except Exception as error:
        print("Voice error:", error)

threading.Thread(target=voice_worker, daemon=True).start()

def speak(message):
    voice_queue.put(message)

# ---------------- YOLO MODEL ----------------

print("Loading human detection model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully.")

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Camera not available. Check camera permissions or try camera index 1."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# ---------------- COUNTERS ----------------

count_in = 0
count_out = 0

# Track the last confirmed side for each person ID.
last_side = {}

# ---------------- MAIN LOOP ----------------

print("Counter started. Press Q or ESC to exit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read from camera.")
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Detect and track people only (COCO class 0 = person).
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            conf=0.30,
            verbose=False
        )

        # Draw the counting line.
        cv2.line(
            frame,
            (LINE_X, 0),
            (LINE_X, HEIGHT),
            (255, 100, 0),
            3
        )

        cv2.putText(
            frame, "OUT", (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255), 3
        )

        cv2.putText(
            frame, "IN", (WIDTH - 100, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 3
        )

        # ---------------- PROCESS PEOPLE ----------------

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                coordinates = boxes.xyxy.cpu().tolist()

                for person_id, box in zip(ids, coordinates):
                    x1, y1, x2, y2 = map(int, box)

                    # Bottom-center of the person box.
                    # This is often more stable than the box center.
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Classify side with a neutral zone around the line.
                    if cx < LINE_X - OFFSET:
                        current_side = "LEFT"
                    elif cx > LINE_X + OFFSET:
                        current_side = "RIGHT"
                    else:
                        current_side = None

                    # First observation: remember the person's side.
                    if person_id not in last_side:
                        if current_side is not None:
                            last_side[person_id] = current_side

                    # Count only when the tracked person changes sides.
                    elif current_side is not None:
                        previous_side = last_side[person_id]

                        if current_side != previous_side:

                            # LEFT -> RIGHT = IN
                            if (
                                previous_side == "LEFT"
                                and current_side == "RIGHT"
                            ):
                                count_in += 1
                                speak("Person entered")
                                print(
                                    f"Person {person_id}: IN = {count_in}"
                                )

                            # RIGHT -> LEFT = OUT
                            elif (
                                previous_side == "RIGHT"
                                and current_side == "LEFT"
                            ):
                                count_out += 1
                                speak("Person exited")
                                print(
                                    f"Person {person_id}: OUT = {count_out}"
                                )

                            last_side[person_id] = current_side

                    # Draw person box and tracking ID.
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                    cv2.circle(
                        frame, (cx, cy), 5,
                        (255, 0, 255), -1
                    )

                    cv2.putText(
                        frame,
                        f"Person {person_id}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

        # ---------------- DISPLAY PANEL ----------------

        inside = max(0, count_in - count_out)

        cv2.rectangle(
            frame, (10, 65), (330, 210),
            (30, 30, 30), -1
        )

        cv2.putText(
            frame, f"IN: {count_in}", (25, 105),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"OUT: {count_out}", (25, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 0, 255), 2
        )

        cv2.putText(
            frame, f"INSIDE: {inside}", (25, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 255, 255), 2
        )

        cv2.imshow("Human IN-OUT Counter", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    voice_queue.put(None)
    print("Counter stopped.")
