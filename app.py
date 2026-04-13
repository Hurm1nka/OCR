# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
import base64
import urllib.request
from paddleocr import PaddleOCR
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 

# =================================================================
# НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app) 

LOG_PATH = os.path.join(BASE_DIR, "ocr_service.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("ocr-service")
logging.getLogger("ppocr").setLevel(logging.ERROR)
logger.info("Service logging initialized. Log file: %s", LOG_PATH)

@app.before_request
def log_request():
    # Короткий лог каждого входящего запроса
    logger.info(
        "REQ %s %s from=%s ct=%s len=%s",
        request.method,
        request.path,
        request.headers.get("X-Forwarded-For", request.remote_addr),
        request.content_type,
        request.content_length,
    )

@app.after_request
def log_response(response):
    logger.info("RES %s %s -> %s", request.method, request.path, response.status)
    return response

print("Загрузка модели PaddleOCR...")
try:
    ocr_reader = PaddleOCR(
        lang="ru",
        use_textline_orientation=False,
        use_gpu=False,
        drop_score=0.3
    )
    print("✅ Модель успешно загружена.")
except Exception as e:
    print(f"❌ ОШИБКА ЗАГРУЗКИ МОДЕЛИ: {e}")
    ocr_reader = None
    logger.exception("PaddleOCR init failed")

VALID_LETTERS = "АВЕКМНОРСТУХ"
VALID_DIGITS = "0123456789"
RUSSIAN_PLATE_CHARS = VALID_LETTERS + VALID_DIGITS

PLATE_REGEX = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}$')
PLATE_WITH_REGION_REGEX = re.compile(r'^([АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2})(\d{2,3})?$')

DIGIT_TO_LETTER = {
    '0': 'О', '4': 'А', '8': 'В', '3': 'З',
}

LETTER_TO_DIGIT = {
    'O': '0', 'О': '0', 'D': '0', 'Q': '0',
    'B': '8', 'В': '8',
    'I': '1', 'L': '1',
    'Z': '2',
    'Т': '7', 'T': '7',
    'S': '5', 'G': '6'
}

LATIN_TO_CYRILLIC = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М', 'H': 'Н', 
    'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т', 'X': 'Х', 'Y': 'У'
}

# =================================================================
# ЛОГИКА КОРРЕКЦИИ
# =================================================================

def normalize_chars(text):
    text = text.upper()
    res = []

    for char in text:
        char = LATIN_TO_CYRILLIC.get(char, char)
        if char in RUSSIAN_PLATE_CHARS:
            res.append(char)

    return "".join(res)

def try_fix_plate(text):
    if len(text) < 6:
        return None

    for i in range(len(text) - 5):
        candidate = list(text[i : i+6])

        for pos in [0, 4, 5]:
            char = candidate[pos]
            if char not in VALID_LETTERS:
                if char in DIGIT_TO_LETTER:
                    candidate[pos] = DIGIT_TO_LETTER[char]

        for pos in [1, 2, 3]:
            char = candidate[pos]
            if char not in VALID_DIGITS:
                if char in LETTER_TO_DIGIT:
                    candidate[pos] = LETTER_TO_DIGIT[char]

        fixed_str = "".join(candidate)

        if (fixed_str[0] in VALID_LETTERS and
            fixed_str[1] in VALID_DIGITS and
            fixed_str[2] in VALID_DIGITS and
            fixed_str[3] in VALID_DIGITS and
            fixed_str[4] in VALID_LETTERS and
            fixed_str[5] in VALID_LETTERS):
            
            return fixed_str

    return None


def try_extract_plate_from_text(text):
    """
    Возвращает номер формата LDDDLL и, при наличии, регион (2-3 цифры).
    """
    if len(text) < 6:
        return None, None

    for i in range(len(text) - 5):
        max_len = min(9, len(text) - i)  # 6 + регион до 3 цифр
        window = text[i : i + max_len]
        fixed_base = try_fix_plate(window[:6])
        if not fixed_base:
            continue

        region = ""
        for ch in window[6:]:
            if ch in VALID_DIGITS:
                region += ch
            elif ch in LETTER_TO_DIGIT:
                region += LETTER_TO_DIGIT[ch]
            else:
                break

        if len(region) not in (0, 2, 3):
            region = ""

        combined = fixed_base + region
        if PLATE_WITH_REGION_REGEX.match(combined):
            return fixed_base, region

    return None, None

# =================================================================
# IMAGE PROCESSING
# =================================================================

def upscale(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))

    if len(coords) < 10:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def generate_image_variants(img):
    variants = []

    img = deskew_image(img)
    img = upscale(img)
    variants.append(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    variants.append(enhanced)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(binary)

    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    variants.append(adaptive)

    adaptive_inv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    variants.append(adaptive_inv)

    return variants

# =================================================================
# OCR PIPELINE
# =================================================================

def process_ocr_pipeline(image_array):
    if ocr_reader is None:
        return "Модель не готова", None, "", 0.0

    variants = generate_image_variants(image_array)

    best_plate = None
    best_region = ""
    best_confidence = 0.0
    logs = []

    for i, img_variant in enumerate(variants):
        try:
            result = ocr_reader.ocr(img_variant, cls=False)

            if not result or not result[0]:
                logs.append(f"Var{i}: OCR не вернул текст")
                continue

            for line in result[0]:
                box = line[0]
                raw_text = line[1][0]
                confidence = line[1][1]

                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]

                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)

                if height == 0:
                    logs.append(f"Var{i}: пропуск бокса (height=0)")
                    continue

                ratio = width / height

                if ratio < 1.2:
                    logs.append(f"Var{i}: '{raw_text}' пропуск ratio={ratio:.2f}")
                    continue

                normalized = normalize_chars(raw_text)
                fixed_plate, region = try_extract_plate_from_text(normalized)
                plate_for_log = (fixed_plate + region) if fixed_plate else None
                log_entry = f"Var{i}: '{raw_text}' -> '{normalized}' -> '{plate_for_log}' ({confidence:.2f})"
                logs.append(log_entry)

                if fixed_plate and PLATE_REGEX.match(fixed_plate):
                    if confidence > best_confidence:
                        best_plate = fixed_plate
                        best_region = region
                        best_confidence = confidence

            if best_plate and best_confidence > 0.9:
                break

        except Exception as e:
            logs.append(f"Var{i}: ошибка OCR {e}")
            continue

    if not best_plate:
        try:
            fallback_result = ocr_reader.ocr(image_array, cls=False)
            if fallback_result and fallback_result[0]:
                merged = "".join([line[1][0] for line in fallback_result[0]])
                normalized = normalize_chars(merged)
                fixed_plate, region = try_extract_plate_from_text(normalized)
                plate_for_log = (fixed_plate + region) if fixed_plate else None
                logs.append(f"Fallback merged: '{merged}' -> '{normalized}' -> '{plate_for_log}'")
                if fixed_plate and PLATE_REGEX.match(fixed_plate):
                    best_plate = fixed_plate
                    best_region = region
                    best_confidence = 0.5
        except Exception as e:
            logs.append(f"Fallback error: {e}")

    explanation = " | ".join(logs[-8:]) if logs else "OCR не нашел подходящих текстовых боксов"
    return explanation, best_plate, best_region, best_confidence


def decode_image_from_bytes(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Ошибка декодирования")
    return img

# =================================================================
# FLASK ROUTES
# =================================================================

@app.route('/')
@app.route('/index.html')
def index_page():
    resp = send_from_directory(BASE_DIR, 'index.html')
    resp.headers['Cache-Control'] = 'no-store, max-age=0, must-revalidate'
    return resp


@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json(silent=True) or {}
    base64_img = data.get('image_data')
    
    if not base64_img:
        logger.warning("process: missing image_data")
        return jsonify({"error": "Нет данных"}), 400

    try:
        logger.info("process: received image_data chars=%s", len(base64_img))
        image_bytes = base64.b64decode(base64_img)
        img = decode_image_from_bytes(image_bytes)

        explanation, plate, region, conf = process_ocr_pipeline(img)
        logger.info("process: plate=%r region=%r conf=%.4f", plate, region, conf)
        full_plate = f"{plate}{region}" if plate else ""

        if plate:
            res = {
                "plate": full_plate,
                "plate_base": plate,
                "region": region,
                "confidence_explanation": f"Найден: {full_plate} ({conf:.2f}). Лог: {explanation}"
            }
        else:
            res = {
                "plate": "",
                "plate_base": "",
                "region": "",
                "confidence_explanation": f"Не найдено. Лог: {explanation}"
            }

        return jsonify(res), 200

    except Exception as e:
        logger.exception("process: failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/process_ip_camera', methods=['POST'])
def process_ip_camera():
    data = request.get_json(silent=True) or {}
    camera_url = (data.get('camera_url') or "").strip()

    if not camera_url:
        return jsonify({"error": "camera_url не указан"}), 400

    try:
        logger.info("process_ip_camera: fetching frame from %s", camera_url)
        req = urllib.request.Request(
            camera_url,
            headers={"User-Agent": "Mozilla/5.0 OCR-Service"}
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            image_bytes = response.read()

        img = decode_image_from_bytes(image_bytes)
        explanation, plate, region, conf = process_ocr_pipeline(img)
        logger.info("process_ip_camera: plate=%r region=%r conf=%.4f", plate, region, conf)
        full_plate = f"{plate}{region}" if plate else ""

        preview_b64 = base64.b64encode(image_bytes).decode("utf-8")
        res = {
            "plate": full_plate,
            "plate_base": plate or "",
            "region": region or "",
            "confidence_explanation": (
                f"Найден: {full_plate} ({conf:.2f}). Лог: {explanation}"
                if plate else
                f"Не найдено. Лог: {explanation}"
            ),
            "preview_data_url": f"data:image/jpeg;base64,{preview_b64}",
        }
        return jsonify(res), 200
    except Exception as e:
        logger.exception("process_ip_camera: failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/process_local_camera', methods=['POST'])
def process_local_camera():
    data = request.get_json(silent=True) or {}
    camera_index = int(data.get('camera_index', 0))

    logger.info("process_local_camera: opening camera index=%s", camera_index)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error("process_local_camera: cannot open camera index=%s", camera_index)
        return jsonify({"error": f"Не удалось открыть локальную камеру index={camera_index}"}), 500

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.error("process_local_camera: failed to read frame index=%s", camera_index)
            return jsonify({"error": "Не удалось получить кадр с локальной камеры"}), 500

        explanation, plate, region, conf = process_ocr_pipeline(frame)
        full_plate = f"{plate}{region}" if plate else ""
        logger.info("process_local_camera: plate=%r region=%r conf=%.4f", plate, region, conf)

        ok_jpg, jpg_buf = cv2.imencode(".jpg", frame)
        preview_data_url = ""
        if ok_jpg:
            preview_b64 = base64.b64encode(jpg_buf.tobytes()).decode("utf-8")
            preview_data_url = f"data:image/jpeg;base64,{preview_b64}"

        return jsonify({
            "plate": full_plate,
            "plate_base": plate or "",
            "region": region or "",
            "confidence_explanation": (
                f"Найден: {full_plate} ({conf:.2f}). Лог: {explanation}"
                if plate else
                f"Не найдено. Лог: {explanation}"
            ),
            "preview_data_url": preview_data_url,
        }), 200
    except Exception as e:
        logger.exception("process_local_camera: failed: %s", e)
        return jsonify({"error": str(e)}), 500
    finally:
        cap.release()

@app.route('/health', methods=['GET'])
def health():
    ready = ocr_reader is not None
    # Важно: фронт сейчас ожидает model_loaded; оставляем совместимость
    return jsonify({"status": "OK", "ready": ready, "model_loaded": ready}), 200

# =================================================================
# RUN
# =================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3020)