# -*- coding: utf-8 -*-
import os
import sys
import re
import time
import ipaddress
import threading
import cv2
import numpy as np
import base64
import urllib.request
from urllib.parse import urlparse
from paddleocr import PaddleOCR
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# =================================================================
# НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app) 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ocr-service")
logging.getLogger("ppocr").setLevel(logging.ERROR)

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

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join(BASE_DIR, "models", "license_plate.pt"))
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
yolo_detector = None
if YOLO is not None and os.path.exists(YOLO_MODEL_PATH):
    try:
        print(f"Загрузка YOLO модели: {YOLO_MODEL_PATH}")
        yolo_detector = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO модель загружена.")
    except Exception as e:
        logger.exception("YOLO init failed: %s", e)
        yolo_detector = None
else:
    logger.info("YOLO detector disabled (module/model missing). path=%s", YOLO_MODEL_PATH)

VALID_LETTERS = "ABEKMHOPCTYX"
VALID_DIGITS = "0123456789"
RUSSIAN_PLATE_CHARS = VALID_LETTERS + VALID_DIGITS

PLATE_REGEX = re.compile(r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}$')
PLATE_WITH_REGION_REGEX = re.compile(r'^([ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2})(\d{2,3})?$')

CYRILLIC_TO_LATIN_PLATE = {
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H',
    'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X',
}

# Сильные/средние/слабые подстановки с разным штрафом.
LETTER_EQUIV_STRONG = {
    '0': ['O'],
    '4': ['A'],
    '8': ['B'],
}
LETTER_EQUIV_MEDIUM = {
    '6': ['B'],
    '9': ['O'],
}

DIGIT_EQUIV_STRONG = {
    'O': ['0'],
    'D': ['0'],
    'Q': ['0'],
    'I': ['1'],
    'L': ['1'],
    'Z': ['2'],
    'S': ['5'],
    'G': ['6'],
    'T': ['7'],
    'B': ['8'],
}
DIGIT_EQUIV_WEAK = {
    'Э': ['3', '9'],
    'З': ['3'],
    'Ч': ['4'],
    'Б': ['6', '8'],
    'Ь': ['6'],
    'Ъ': ['6'],
    'Ђ': ['6', '5', '9'],
    'Ћ': ['6', '5', '9'],
    'Є': ['6'],
}

ALLOWED_EXTRA_CHARS = set(DIGIT_EQUIV_WEAK.keys())

DEDUP_WINDOW_SECONDS = int(os.getenv("DEDUP_WINDOW_SECONDS", "30"))
recent_plate_events = {}
recent_plate_events_lock = threading.Lock()

# =================================================================
# ЛОГИКА КОРРЕКЦИИ
# =================================================================

def normalize_chars(text):
    text = text.upper()
    res = []
    for char in text:
        if 'A' <= char <= 'Z' or '0' <= char <= '9':
            res.append(char)
            continue
        mapped = CYRILLIC_TO_LATIN_PLATE.get(char)
        if mapped:
            res.append(mapped)
            continue
        if char in ALLOWED_EXTRA_CHARS:
            res.append(char)
    normalized = "".join(res)
    # Хвост региона "RUS" не участвует в маске.
    normalized = normalized.replace("RUS", "")
    return normalized

def letter_options(char):
    c = CYRILLIC_TO_LATIN_PLATE.get(char, char)
    opts = []
    if c in VALID_LETTERS:
        opts.append((c, 0.0))
    for cand in LETTER_EQUIV_STRONG.get(c, []):
        opts.append((cand, 0.18))
    for cand in LETTER_EQUIV_MEDIUM.get(c, []):
        opts.append((cand, 0.35))
    # Убираем дубликаты, оставляя минимальный штраф
    best = {}
    for cand, cost in opts:
        if cand in VALID_LETTERS and (cand not in best or cost < best[cand]):
            best[cand] = cost
    return list(best.items())


def digit_options(char):
    c = CYRILLIC_TO_LATIN_PLATE.get(char, char)
    opts = []
    if c in VALID_DIGITS:
        opts.append((c, 0.0))
    for cand in DIGIT_EQUIV_STRONG.get(c, []):
        opts.append((cand, 0.18))
    for cand in DIGIT_EQUIV_WEAK.get(c, []):
        opts.append((cand, 0.55))
    best = {}
    for cand, cost in opts:
        if cand in VALID_DIGITS and (cand not in best or cost < best[cand]):
            best[cand] = cost
    return list(best.items())


def try_fix_plate_from_window(window6):
    if len(window6) < 6:
        return None, 999.0
    chars = list(window6[:6])
    pos_candidates = []
    for idx, ch in enumerate(chars):
        opts = letter_options(ch) if idx in (0, 4, 5) else digit_options(ch)
        if not opts:
            return None, 999.0
        pos_candidates.append(opts[:5])

    best_candidate = None
    best_cost = 999.0
    for c0, k0 in pos_candidates[0]:
        for c1, k1 in pos_candidates[1]:
            for c2, k2 in pos_candidates[2]:
                for c3, k3 in pos_candidates[3]:
                    for c4, k4 in pos_candidates[4]:
                        for c5, k5 in pos_candidates[5]:
                            candidate = f"{c0}{c1}{c2}{c3}{c4}{c5}"
                            if PLATE_REGEX.match(candidate):
                                cost = k0 + k1 + k2 + k3 + k4 + k5
                                if cost < best_cost:
                                    best_cost = cost
                                    best_candidate = candidate
    return best_candidate, best_cost


def try_fix_plate(text):
    if len(text) < 6:
        return None, 999.0
    best_candidate = None
    best_cost = 999.0
    for i in range(len(text) - 5):
        fixed, cost = try_fix_plate_from_window(text[i : i + 6])
        if fixed and cost < best_cost:
            best_candidate = fixed
            best_cost = cost
    return best_candidate, best_cost


def try_extract_plate_from_text(text):
    if len(text) < 6:
        return None, None, 999.0
    best_plate = None
    best_region = ""
    best_cost = 999.0
    for i in range(len(text) - 5):
        max_len = min(9, len(text) - i)
        window = text[i : i + max_len]
        fixed_base, base_cost = try_fix_plate(window[:6])
        if not fixed_base:
            continue

        region_chars = []
        region_cost = 0.0
        for ch in window[6:9]:
            opts = digit_options(ch)
            if not opts:
                break
            best_digit, best_digit_cost = min(opts, key=lambda x: x[1])
            region_chars.append(best_digit)
            region_cost += best_digit_cost
        region = "".join(region_chars)
        if len(region) not in (0, 2, 3):
            region = ""
            region_cost = 0.0
        combined = f"{fixed_base}{region}"
        if PLATE_WITH_REGION_REGEX.match(combined):
            total_cost = base_cost + region_cost
            if total_cost < best_cost:
                best_plate = fixed_base
                best_region = region
                best_cost = total_cost
    return best_plate, best_region, best_cost


def prune_recent_plate_events(now_ts):
    with recent_plate_events_lock:
        stale = [k for k, ts in recent_plate_events.items() if now_ts - ts > DEDUP_WINDOW_SECONDS]
        for key in stale:
            del recent_plate_events[key]


def mark_plate_event_and_check_duplicate(plate_base):
    now_ts = time.time()
    prune_recent_plate_events(now_ts)
    with recent_plate_events_lock:
        prev_ts = recent_plate_events.get(plate_base)
        is_duplicate_recent = prev_ts is not None and (now_ts - prev_ts) <= DEDUP_WINDOW_SECONDS
        recent_plate_events[plate_base] = now_ts
    return is_duplicate_recent

# =================================================================
# IMAGE PROCESSING
# =================================================================

def upscale(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def upscale_adaptive(img):
    h, w = img.shape[:2]
    # Для маленьких кадров агрессивнее увеличиваем изображение.
    factor = 3 if max(h, w) < 900 else 2
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

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
    img = upscale_adaptive(img)
    variants.append(img)
    variants.append(sharpen_image(img))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_sharp = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray_sharp = cv2.addWeighted(gray, 1.6, gray_sharp, -0.6, 0)
    variants.append(gray_sharp)

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


def extract_plate_like_regions(img):
    """
    Извлекает ROI, похожие на номер (широкие прямоугольники с высокой контрастностью).
    Это помогает OCR сфокусироваться на табличке, а не на всем кадре.
    """
    rois = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    _, bw = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape[:2]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue
        ratio = w / float(h)
        area = w * h
        # Типичный диапазон пропорций/площади для номерной таблички в кадре
        if ratio < 2.0 or ratio > 7.0:
            continue
        if area < 0.01 * w_img * h_img:
            continue
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            rois.append(roi)

    return rois[:5]


def extract_plate_regions_yolo(img):
    """
    Поиск номера через YOLO-детектор.
    Возвращает crops и короткие логи.
    """
    if yolo_detector is None:
        return [], ["YOLO: detector disabled"]
    try:
        results = yolo_detector.predict(
            source=img,
            conf=YOLO_CONF_THRESHOLD,
            verbose=False,
            device="cpu",
            imgsz=960,
        )
        if not results:
            return [], ["YOLO: no results"]
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return [], ["YOLO: no boxes"]

        h, w = img.shape[:2]
        crops = []
        logs = [f"YOLO: boxes={len(r.boxes)} conf>={YOLO_CONF_THRESHOLD:.2f}"]
        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        boxes_conf = r.boxes.conf.cpu().numpy()
        order = np.argsort(-boxes_conf)[:3]
        for idx in order:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            conf = float(boxes_conf[idx])
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad_x = int(bw * 0.08)
            pad_y = int(bh * 0.20)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                logs.append(f"YOLO box conf={conf:.2f} size={x2-x1}x{y2-y1}")
        return crops, logs
    except Exception as e:
        return [], [f"YOLO error: {e}"]

# =================================================================
# OCR PIPELINE
# =================================================================

def process_ocr_pipeline(image_array):
    if ocr_reader is None:
        return "Модель не готова", "OCR pipeline не запущен: модель не готова", None, "", 0.0

    yolo_rois, yolo_logs = extract_plate_regions_yolo(image_array)
    variants = []
    for roi in yolo_rois:
        variants.extend(generate_image_variants(roi))
    variants.extend(generate_image_variants(image_array))
    roi_variants = []
    for roi in extract_plate_like_regions(image_array):
        roi_variants.extend(generate_image_variants(roi))
    variants = variants + roi_variants

    best_plate = None
    best_region = ""
    best_confidence = 0.0
    best_total_score = -999.0
    logs = list(yolo_logs)

    for i, img_variant in enumerate(variants):
        try:
            result = ocr_reader.ocr(img_variant, cls=False)

            if not result or not result[0]:
                logs.append(f"Var{i}: OCR не вернул текст")
                continue

            recognized_chunks = []
            for line in result[0]:
                box = line[0]
                raw_text = line[1][0]
                confidence = line[1][1]
                recognized_chunks.append((box, raw_text, confidence))

                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]

                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)

                if height == 0:
                    logs.append(f"Var{i}: пропуск бокса (height=0)")
                    continue

                ratio = width / height

                if ratio < 0.9:
                    logs.append(f"Var{i}: '{raw_text}' пропуск ratio={ratio:.2f}")
                    continue

                normalized = normalize_chars(raw_text)
                fixed_plate, region, correction_cost = try_extract_plate_from_text(normalized)
                plate_for_log = (fixed_plate + region) if fixed_plate else None
                log_entry = f"Var{i}: '{raw_text}' -> '{normalized}' -> '{plate_for_log}' ({confidence:.2f}, cost={correction_cost:.2f})"
                logs.append(log_entry)

                if fixed_plate and PLATE_REGEX.match(fixed_plate):
                    # Совмещаем уверенность OCR и штраф за агрессивные подстановки.
                    total_score = confidence - (0.22 * correction_cost)
                    if total_score > best_total_score:
                        best_plate = fixed_plate
                        best_region = region
                        best_confidence = confidence
                        best_total_score = total_score

            # Частый кейс: OCR вернул номер в нескольких кусках (например "A695KA" + "799").
            if recognized_chunks:
                by_position = sorted(
                    recognized_chunks,
                    key=lambda item: (min(p[1] for p in item[0]), min(p[0] for p in item[0]))
                )
                merged_raw = "".join([item[1] for item in by_position])
                merged_norm = normalize_chars(merged_raw)
                fixed_plate, region, correction_cost = try_extract_plate_from_text(merged_norm)
                merged_conf = float(np.mean([item[2] for item in by_position]))
                merged_log = f"Var{i} merged: '{merged_raw}' -> '{merged_norm}' -> '{(fixed_plate + region) if fixed_plate else None}' ({merged_conf:.2f}, cost={correction_cost:.2f})"
                logs.append(merged_log)
                if fixed_plate and PLATE_REGEX.match(fixed_plate):
                    total_score = merged_conf - (0.22 * correction_cost) + 0.05
                    if total_score > best_total_score:
                        best_plate = fixed_plate
                        best_region = region
                        best_confidence = merged_conf
                        best_total_score = total_score

            if best_plate and best_total_score > 0.88:
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
                fixed_plate, region, correction_cost = try_extract_plate_from_text(normalized)
                plate_for_log = (fixed_plate + region) if fixed_plate else None
                logs.append(f"Fallback merged: '{merged}' -> '{normalized}' -> '{plate_for_log}' (cost={correction_cost:.2f})")
                if fixed_plate and PLATE_REGEX.match(fixed_plate):
                    best_plate = fixed_plate
                    best_region = region
                    best_confidence = 0.5
                    best_total_score = 0.5 - (0.22 * correction_cost)
        except Exception as e:
            logs.append(f"Fallback error: {e}")

    ocr_log = " | ".join(logs[-8:]) if logs else "OCR не нашел подходящих текстовых боксов"
    if best_plate:
        model_explanation = f"Обнаружен шаблон номера. Уверенность OCR: {best_confidence:.2f}."
    else:
        model_explanation = "Шаблон номера РФ не обнаружен."
    return model_explanation, ocr_log, best_plate, best_region, best_confidence


def decode_image_from_bytes(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Ошибка декодирования")
    return img


def normalize_camera_url(url):
    """
    Убирает типичные опечатки: пробелы между хостом и портом
    (например «rtsp://192.168.1.5  8556 /path» → «rtsp://192.168.1.5:8556/path»).
    """
    url = (url or "").strip()
    if not url:
        return url
    url = re.sub(
        r"(?i)((?:rtsp|rtsps|https?)://[^\s/]+)\s+(\d{2,5})",
        r"\1:\2",
        url,
        count=1,
    )
    url = re.sub(r"\s+", "", url)
    return url


def _hint_if_camera_host_unreachable_from_internet(camera_url):
    """Частная сеть (192.168.x и т.д.) с удалённого VPS не маршрутизируется."""
    host = urlparse(camera_url).hostname
    if not host:
        return ""
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_link_local:
            return (
                " Камера по адресу в частной сети (192.168.x.x, 10.x, 172.16–31.x) с **удалённого сервера** "
                "(VPS в интернете) недоступна: пакеты туда не дойдут. Запускайте OCR **на ПК в той же LAN**, "
                "что и камера, или пробросьте поток наружу (VPN, FRP, облачный relay). "
                "Для IP Webcam чаще используйте **HTTP**: http://IP:8080/shot.jpg"
            )
    except ValueError:
        pass
    return ""


def fetch_image_for_ip_camera(camera_url):
    """
    HTTP(S): скачивает байты картинки (как snapshot IP Webcam: /shot.jpg).
    RTSP: один кадр через OpenCV+FFmpeg (urllib не поддерживает rtsp://).
    Возвращает (img_bgr, preview_jpeg_bytes).
    """
    u = camera_url.strip()
    low = u.lower()
    if low.startswith(("rtsp://", "rtsps://")):
        # TCP чаще проходит NAT/фаерволы, чем UDP по умолчанию у RTSP.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

        cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            hint = _hint_if_camera_host_unreachable_from_internet(u)
            raise RuntimeError(
                "Не удалось открыть RTSP (OpenCV/FFmpeg). Проверьте URL, логин/пароль, порт и что поток "
                "существует. Попробуйте открыть тот же URL в VLC."
                + hint
                + " Для снимка с телефона (IP Webcam) обычно нужен HTTP: http://IP:8080/shot.jpg"
            )
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        frame = None
        try:
            for _ in range(20):
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    break
                time.sleep(0.08)
        finally:
            cap.release()
        if frame is None or not getattr(frame, "size", 0):
            raise RuntimeError(
                "RTSP: не удалось прочитать кадр (поток пуст, неверный путь или кодек)."
                + _hint_if_camera_host_unreachable_from_internet(u)
            )
        ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok_jpg:
            raise RuntimeError("Не удалось закодировать кадр в JPEG.")
        return frame, buf.tobytes()

    req = urllib.request.Request(
        u,
        headers={"User-Agent": "Mozilla/5.0 OCR-Service"},
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        image_bytes = response.read()
    img = decode_image_from_bytes(image_bytes)
    ok_jpg, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    preview = buf.tobytes() if ok_jpg else image_bytes
    return img, preview


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

        model_explanation, ocr_log, plate, region, conf = process_ocr_pipeline(img)
        logger.info("process: plate=%r region=%r conf=%.4f", plate, region, conf)
        full_plate = f"{plate}{region}" if plate else ""
        is_duplicate_recent = mark_plate_event_and_check_duplicate(plate) if plate else False

        if plate:
            res = {
                "plate": plate,
                "plate_base": plate,
                "plate_full": full_plate,
                "region": region,
                "confidence_explanation": f"Найден: {full_plate} ({conf:.2f}).",
                "model_explanation": model_explanation,
                "ocr_log": ocr_log,
                "is_duplicate_recent": is_duplicate_recent,
                "dedup_window_seconds": DEDUP_WINDOW_SECONDS,
            }
        else:
            res = {
                "plate": "",
                "plate_base": "",
                "plate_full": "",
                "region": "",
                "confidence_explanation": "Не найдено.",
                "model_explanation": model_explanation,
                "ocr_log": ocr_log,
                "is_duplicate_recent": False,
                "dedup_window_seconds": DEDUP_WINDOW_SECONDS,
            }

        return jsonify(res), 200

    except Exception as e:
        logger.exception("process: failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/process_ip_camera', methods=['POST'])
def process_ip_camera():
    data = request.get_json(silent=True) or {}
    camera_url = normalize_camera_url((data.get("camera_url") or "").strip())

    if not camera_url:
        return jsonify({"error": "camera_url не указан"}), 400

    try:
        logger.info("process_ip_camera: fetching frame from %s", camera_url)
        img, preview_bytes = fetch_image_for_ip_camera(camera_url)
        model_explanation, ocr_log, plate, region, conf = process_ocr_pipeline(img)
        logger.info("process_ip_camera: plate=%r region=%r conf=%.4f", plate, region, conf)
        full_plate = f"{plate}{region}" if plate else ""
        is_duplicate_recent = mark_plate_event_and_check_duplicate(plate) if plate else False

        preview_b64 = base64.b64encode(preview_bytes).decode("utf-8")
        res = {
            "plate": plate or "",
            "plate_base": plate or "",
            "plate_full": full_plate if plate else "",
            "region": region or "",
            "confidence_explanation": (f"Найден: {full_plate} ({conf:.2f})." if plate else "Не найдено."),
            "model_explanation": model_explanation,
            "ocr_log": ocr_log,
            "preview_data_url": f"data:image/jpeg;base64,{preview_b64}",
            "is_duplicate_recent": is_duplicate_recent,
            "dedup_window_seconds": DEDUP_WINDOW_SECONDS,
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
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("process_local_camera: cannot open camera index=%s", camera_index)
        return jsonify({"error": f"Не удалось открыть локальную камеру index={camera_index}"}), 500

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.error("process_local_camera: failed to read frame index=%s", camera_index)
            return jsonify({"error": "Не удалось получить кадр с локальной камеры"}), 500

        model_explanation, ocr_log, plate, region, conf = process_ocr_pipeline(frame)
        full_plate = f"{plate}{region}" if plate else ""
        is_duplicate_recent = mark_plate_event_and_check_duplicate(plate) if plate else False
        logger.info("process_local_camera: plate=%r region=%r conf=%.4f", plate, region, conf)

        ok_jpg, jpg_buf = cv2.imencode(".jpg", frame)
        preview_data_url = ""
        if ok_jpg:
            preview_b64 = base64.b64encode(jpg_buf.tobytes()).decode("utf-8")
            preview_data_url = f"data:image/jpeg;base64,{preview_b64}"

        return jsonify({
            "plate": plate or "",
            "plate_base": plate or "",
            "plate_full": full_plate if plate else "",
            "region": region or "",
            "confidence_explanation": (f"Найден: {full_plate} ({conf:.2f})." if plate else "Не найдено."),
            "model_explanation": model_explanation,
            "ocr_log": ocr_log,
            "preview_data_url": preview_data_url,
            "is_duplicate_recent": is_duplicate_recent,
            "dedup_window_seconds": DEDUP_WINDOW_SECONDS,
        }), 200
    except Exception as e:
        logger.exception("process_local_camera: failed: %s", e)
        return jsonify({"error": str(e)}), 500
    finally:
        cap.release()

@app.route('/health', methods=['GET'])
def health():
    ready = ocr_reader is not None
    yolo_ready = yolo_detector is not None
    # Важно: фронт сейчас ожидает model_loaded; оставляем совместимость
    return jsonify({
        "status": "OK",
        "ready": ready,
        "model_loaded": ready,
        "yolo_loaded": yolo_ready,
        "yolo_model_path": YOLO_MODEL_PATH,
    }), 200


@app.route('/barrier/open', methods=['POST'])
def barrier_open():
    """
    Заглушка ручного открытия шлагбаума.
    В будущем сюда можно подключить GPIO/PLC/внешний контроллер.
    """
    payload = request.get_json(silent=True) or {}
    reason = (payload.get("reason") or "manual").strip()[:64]
    logger.info("barrier_open: stub triggered reason=%s", reason)
    return jsonify({
        "ok": True,
        "message": "Команда открытия шлагбаума отправлена (заглушка).",
        "reason": reason,
        "opened_at": int(time.time()),
    }), 200

# =================================================================
# RUN
# =================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3020)