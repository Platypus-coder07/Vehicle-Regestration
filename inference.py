import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

if __name__ == '__main__':

    # ── CONFIG ──────────────────────────────────────────────────────
    MODEL_PATH  = "saved_models/license_plate_best.pt"
    VIDEO_PATH  = "input_video.mp4"   # put your video here
    OUTPUT_PATH = "output_video.mp4"
    CONF_THRESH = 0.4
    WINDOW_SIZE = 30
    # ────────────────────────────────────────────────────────────────

    # Load models
    model  = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=True)

    # Majority voting buffer
    plate_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

    def preprocess_plate(img):
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        enhanced = cv2.fastNlMeansDenoising(enhanced, h=10)
        return enhanced

    def read_plate_text(plate_img):
        processed = preprocess_plate(plate_img)
        results   = reader.readtext(processed, detail=0, paragraph=False)
        if results:
            return results[0].upper().replace(" ", "").replace("-", "")
        return None

    def get_majority_vote(history_deque):
        if not history_deque:
            return None
        counter = defaultdict(int)
        for r in history_deque:
            counter[r] += 1
        return max(counter, key=counter.get)

    # ── VIDEO LOOP ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=CONF_THRESH, verbose=False)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else 0

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                plate_text = read_plate_text(plate_crop)
                if plate_text:
                    plate_history[track_id].append(plate_text)

                stable_text = get_majority_vote(plate_history[track_id])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw plate text
                if stable_text:
                    cv2.putText(frame, stable_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Zoomed plate preview top-left
                if plate_crop.size > 0:
                    zoomed = cv2.resize(plate_crop, (200, 80))
                    frame[10:90, 10:210] = zoomed

        out.write(frame)
        cv2.imshow("License Plate Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done! Output saved to:", OUTPUT_PATH)