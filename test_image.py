import cv2
import easyocr
from ultralytics import YOLO
import re

model  = YOLO("saved_models/license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# ── Indian Plate Formats ─────────────────────────────────────────────────────
PLATE_PATTERNS = [
    # Standard: MH12AB1234
    (re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'), "Standard Indian"),
    # BH Series: 22BH6517A
    (re.compile(r'^[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]$'),         "BH Series"),
    # Diplomatic: 77CD1234
    (re.compile(r'^[0-9]{2}[A-Z]{2}[0-9]{4}$'),              "Diplomatic"),
]

def preprocess_plate(img):
    img       = cv2.resize(img, (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced  = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised  = cv2.fastNlMeansDenoising(thresh, h=10)
    return denoised

def apply_cci(text):
    """
    Check Character Index — correct common OCR mistakes based on
    expected character type at each position for Indian plate formats.
    
    Common confusions:
      O <-> 0    I <-> 1    S <-> 5
      B <-> 8    Z <-> 2    G <-> 6
    """
    CHAR_TO_NUM = {'O':'0', 'I':'1', 'S':'5', 'B':'8', 'Z':'2', 'G':'6', 'D':'0'}
    NUM_TO_CHAR = {'0':'O', '1':'I', '5':'S', '8':'B', '2':'Z', '6':'G'}

    text = text.upper().replace(" ", "")

    # Try BH series: NN LL NNNN L  (e.g. 22BH6517A)
    if len(text) == 9:
        corrected = ""
        for i, ch in enumerate(text):
            if i in [0, 1, 4, 5, 6, 7]:   # should be digit
                corrected += CHAR_TO_NUM.get(ch, ch)
            elif i in [2, 3, 8]:            # should be letter
                corrected += NUM_TO_CHAR.get(ch, ch)
        return corrected

    # Try Standard: LL NN LLL NNNN (e.g. MH12AB1234)
    if len(text) == 10:
        corrected = ""
        for i, ch in enumerate(text):
            if i in [0, 1, 4, 5, 6]:       # should be letter
                corrected += NUM_TO_CHAR.get(ch, ch)
            elif i in [2, 3, 7, 8, 9]:      # should be digit
                corrected += CHAR_TO_NUM.get(ch, ch)
        return corrected

    return text  # Unknown format, return as-is

def smart_join_ocr(ocr_results):
    """Join multiple OCR tokens, skip country codes, apply CCI."""
    # Sort by x-position (left to right)
    sorted_results = sorted(ocr_results, key=lambda x: x[0][0][0])

    tokens = []
    for (bbox, text, conf) in sorted_results:
        cleaned = text.upper().replace(" ", "").replace("-", "")
        # Skip country codes and very short tokens
        if cleaned in ['IND', 'EU', 'UK', 'USA'] or len(cleaned) < 2:
            continue
        tokens.append((cleaned, conf))

    if not tokens:
        return "N/A", 0

    # Join all tokens
    joined     = "".join(t[0] for t in tokens)
    avg_conf   = sum(t[1] for t in tokens) / len(tokens)

    # Apply CCI correction
    corrected  = apply_cci(joined)

    return corrected, avg_conf

def validate_plate(text):
    """Check if plate matches any known Indian format."""
    for pattern, name in PLATE_PATTERNS:
        if pattern.match(text):
            return True, name
    return False, "Unknown format"

def test_image(image_path):
    img     = cv2.imread(image_path)
    results = model(img, conf=0.3)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        print("No license plate detected!")
        return

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf       = float(box.conf[0])
        plate_crop = img[y1:y2, x1:x2]

        processed  = preprocess_plate(plate_crop)
        ocr_result = reader.readtext(processed, detail=1,
                                     allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        plate_text, ocr_conf = smart_join_ocr(ocr_result)
        is_valid, fmt_name   = validate_plate(plate_text)

        print(f"\n--- Plate {i+1} ---")
        print(f"YOLO Confidence : {conf:.2f}")
        print(f"Raw OCR tokens  : {[(t, f'{c:.2f}') for _,t,c in ocr_result]}")
        print(f"Joined + CCI    : {plate_text}")
        print(f"OCR Confidence  : {ocr_conf:.2f}")
        print(f"Format valid    : {'✓ ' + fmt_name if is_valid else '✗ Unknown'}")

        color = (0, 255, 0) if is_valid else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, plate_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Plate - Original",     plate_crop)
        cv2.imshow("Plate - Preprocessed", processed)

    cv2.imshow("Full Image Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# test_image("test_car.jpg")

test_image("test_car_1.webp")   # change to your image name