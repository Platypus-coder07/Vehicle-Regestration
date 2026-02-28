import cv2
import os
import re
import google.generativeai as genai
from PIL import Image
from ultralytics import YOLO

# ── Load YOLO ─────────────────────────────────────────────────────────────────
model = YOLO("saved_models/license_plate_best.pt")

IMAGES_FOLDER = "License-Plate-Recognition-4/test/images"
API_KEY       = ""  # Paste your key here

# ── Configure Gemini ──────────────────────────────────────────────────────────
genai.configure(api_key=API_KEY)

def get_working_model():
    """
    Automatically finds a working Vision model available to your API key.
    Tries 1.5-Flash first (fastest), then falls back to others.
    """
    preferred_models = [
        'gemini-1.5-flash',
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash-001',
        'gemini-1.5-pro',
        'gemini-pro-vision'  # Legacy fallback
    ]
    
    print("Checking available Gemini models...")
    try:
        # List models available to this API key
        available = [m.name.replace('models/', '') for m in genai.list_models()]
        
        # 1. Check if any of our preferred models exist in the user's list
        for model_name in preferred_models:
            if model_name in available:
                print(f"✓ Found supported model: {model_name}")
                return genai.GenerativeModel(model_name)
        
        # 2. If we couldn't match, force try the first preferred one
        print(f"! Could not list specific models. Defaulting to: {preferred_models[0]}")
        return genai.GenerativeModel(preferred_models[0])
        
    except Exception as e:
        print(f"! Model listing failed ({e}). Defaulting to: {preferred_models[0]}")
        return genai.GenerativeModel(preferred_models[0])

# Initialize the best model found
gemini_model = get_working_model()

# ── Plate Format Definitions ──────────────────────────────────────────────────
PLATE_FORMATS = [
    (re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'), "Indian Standard", "IN"),
    (re.compile(r'^[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]$'),         "Indian BH",       "IN"),
    (re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$'),              "UK Standard",     "UK"),
    (re.compile(r'^[A-Z][0-9]{1,3}[A-Z]{3}$'),               "UK Prefix",       "UK"),
    (re.compile(r'^[0-9]{2}[A-Z]{1,2}[0-9]{4,5}$'),          "Vietnamese Car",  "VN"),
    (re.compile(r'^[A-Z]{2,3}[0-9]{2,3}[A-Z]{0,2}$'),        "Australian",      "AU"),
    (re.compile(r'^[A-Z0-9]{5,8}$'),                          "Generic",         "??"),
]

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_plate(img):
    img       = cv2.resize(img, (0,0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced  = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised  = cv2.fastNlMeansDenoising(thresh, h=10)
    return denoised

# ── Gemini Vision OCR ─────────────────────────────────────────────────────────
def cv2_to_pil(cv2_img):
    """Convert OpenCV BGR image to PIL RGB image."""
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def gemini_read_plate(plate_img):
    """Send plate image to Gemini and extract text."""
    # Preprocess
    processed = preprocess_plate(plate_img)
    
    # Convert to PIL images (Gemini preferred format)
    pil_original  = cv2_to_pil(plate_img)
    pil_processed = cv2_to_pil(processed)

    prompt = (
        "These are two versions of the same license plate image "
        "(original and preprocessed). "
        "Read the license plate number/text exactly as it appears. "
        "Reply with ONLY the plate text, no spaces, no punctuation, "
        "no explanation. Example: MH12AB1234"
    )

    try:
        # Generate content with prompt and images
        response = gemini_model.generate_content([prompt, pil_original, pil_processed])
        
        # Check if response contains text
        if response.text:
            raw = response.text.strip()
            # Clean up response — remove any spaces/punctuation
            cleaned = re.sub(r'[^A-Z0-9]', '', raw.upper())
            return cleaned
        else:
            return "N/A"

    except Exception as e:
        print(f"   Request failed: {e}")
        return "N/A"

# ── Format Validation ─────────────────────────────────────────────────────────
def validate_plate(text):
    for pattern, name, country in PLATE_FORMATS:
        if pattern.match(text):
            return True, name, country
    return False, "Unknown", "??"

# ── Main Batch Test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    
    # Check if folder exists
    if not os.path.exists(IMAGES_FOLDER):
        print(f"Error: Folder '{IMAGES_FOLDER}' not found.")
        exit()

    image_files = sorted([
        f for f in os.listdir(IMAGES_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    total          = len(image_files)
    detected       = 0
    valid_fmt      = 0
    no_plate       = 0
    country_counts = {}

    print(f"Found {total} test images")
    print("Using Gemini Vision for OCR")
    print("=" * 60)
    print("Controls: any key = next | Q = quit")
    print("=" * 60)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(IMAGES_FOLDER, image_file)
        img        = cv2.imread(image_path)

        if img is None:
            continue

        results = model(img, conf=0.3, verbose=False)
        print(f"\n[{idx+1}/{total}] {image_file}")

        if results[0].boxes is None or len(results[0].boxes) == 0:
            print("   ✗ No plate detected")
            no_plate += 1
            cv2.imshow("Batch Test (any key=next | Q=quit)", img)
            key = cv2.waitKey(0)
            if key == ord('q') or key == ord('Q'):
                break
            continue

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_conf  = float(box.conf[0])
            plate_crop = img[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            detected += 1

            # Gemini Vision OCR
            print(f"   Sending to Gemini Vision...")
            plate_text = gemini_read_plate(plate_crop)

            is_valid, fmt_name, country = validate_plate(plate_text)

            if is_valid:
                valid_fmt += 1
                country_counts[country] = country_counts.get(country, 0) + 1

            print(f"   YOLO conf   : {yolo_conf:.2f}")
            print(f"   Plate text  : {plate_text}")
            print(f"   Format      : {'✓ ' + fmt_name + ' [' + country + ']' if is_valid else '✗ Unknown'}")

            color = (0, 255, 0) if is_valid else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(img, fmt_name + " [" + country + "]", (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            try:
                zoom = cv2.resize(plate_crop, (240, 80))
                img[10:90, 10:250] = zoom
            except:
                pass

        cv2.imshow("Batch Test (any key=next | Q=quit)", img)
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            print("\nStopped by user.")
            break

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("BATCH TEST SUMMARY")
    print("=" * 60)
    print(f"Total images       : {total}")
    print(f"Plates detected    : {detected}")
    print(f"No plate found     : {no_plate}")
    print(f"Valid format       : {valid_fmt}")
    print(f"Detection rate     : {detected/total*100:.1f}%")
    if detected > 0:
        print(f"Format match rate  : {valid_fmt/detected*100:.1f}%")
    if country_counts:
        print(f"\nBy country:")
        for c, count in sorted(country_counts.items(), key=lambda x: -x[1]):
            print(f"   {c} : {count}")
    print("=" * 60)