# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
import base64
from paddleocr import PaddleOCR
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS 

# =================================================================
# НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# =================================================================
app = Flask(__name__)
CORS(app) 

logging.getLogger("ppocr").setLevel(logging.ERROR) 

print("Загрузка модели PaddleOCR...")
try:
    # Используем русскую модель. 
    # drop_score=0.3 позволяет находить даже неуверенные результаты (мы их отфильтруем сами)
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

# 12 букв, используемых в номерах РФ (кириллица)
VALID_LETTERS = "АВЕКМНОРСТУХ"
VALID_DIGITS = "0123456789"
RUSSIAN_PLATE_CHARS = VALID_LETTERS + VALID_DIGITS

# Словари для исправления частых ошибок OCR
# Если на месте БУКВЫ стоит ЦИФРА (или похожий символ), меняем:
DIGIT_TO_LETTER = {
    '0': 'О', '4': 'А', '8': 'В', '3': 'З', # З - не валидна, но часто путается с 3
}
# Если на месте ЦИФРЫ стоит БУКВА, меняем:
LETTER_TO_DIGIT = {
    'O': '0', 'О': '0', 'D': '0', 'Q': '0',
    'B': '8', 'В': '8',
    'I': '1', 'L': '1',
    'Z': '7',
    'S': '5', 'G': '6'
}
# Общая карта замены латиницы на кириллицу
LATIN_TO_CYRILLIC = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М', 'H': 'Н', 
    'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т', 'X': 'Х', 'Y': 'У'
}

# =================================================================
# ЛОГИКА КОРРЕКЦИИ (МОЗГИ)
# =================================================================

def normalize_chars(text):
    """Первичная очистка: латиница -> кириллица, удаление мусора."""
    text = text.upper()
    res = []
    for char in text:
        # Сначала меняем латиницу на кириллицу
        char = LATIN_TO_CYRILLIC.get(char, char)
        # Оставляем только если это валидный символ для номера
        if char in RUSSIAN_PLATE_CHARS:
            res.append(char)
    return "".join(res)

def try_fix_plate(text):
    """
    Пытается превратить строку в формат L DDD LL (6 символов).
    Принимает строку любой длины, ищет в ней 6 символов.
    """
    if len(text) < 6:
        return None

    # Проходим скользящим окном по 6 символов
    for i in range(len(text) - 5):
        candidate = list(text[i : i+6])
        
        # Структура номера РФ: 
        # Позиции: 0 (Буква), 1-3 (Цифры), 4-5 (Буквы)
        
        # --- ШАГ 1: Проверка и исправление БУКВ (поз 0, 4, 5) ---
        for pos in [0, 4, 5]:
            char = candidate[pos]
            if char not in VALID_LETTERS:
                # Если это не буква, пробуем исправить цифру в букву
                if char in DIGIT_TO_LETTER:
                    candidate[pos] = DIGIT_TO_LETTER[char]
        
        # --- ШАГ 2: Проверка и исправление ЦИФР (поз 1, 2, 3) ---
        for pos in [1, 2, 3]:
            char = candidate[pos]
            if char not in VALID_DIGITS:
                # Если это не цифра, пробуем исправить букву в цифру
                if char in LETTER_TO_DIGIT:
                    candidate[pos] = LETTER_TO_DIGIT[char]

        # --- ШАГ 3: Финальная валидация ---
        # Проверяем, получился ли валидный шаблон
        fixed_str = "".join(candidate)
        
        # Проверка: L DDD LL
        if (fixed_str[0] in VALID_LETTERS and
            fixed_str[1] in VALID_DIGITS and
            fixed_str[2] in VALID_DIGITS and
            fixed_str[3] in VALID_DIGITS and
            fixed_str[4] in VALID_LETTERS and
            fixed_str[5] in VALID_LETTERS):
            
            return fixed_str # УСПЕХ!

    return None

# =================================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЙ
# =================================================================

def generate_image_variants(img):
    """Генерирует варианты изображения для улучшения шансов OCR."""
    variants = []
    
    # 1. Оригинал
    variants.append(img)
    
    # 2. Оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(gray)
    
    # 3. Усиленный контраст (CLAHE) - спасает в темноте
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    variants.append(enhanced)
    
    # 4. Бинаризация (Ч/Б) - спасает при шуме
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary)
    
    return variants

def process_ocr_pipeline(image_array):
    """Главный конвейер обработки."""
    if ocr_reader is None:
        return "Модель не готова", None, 0.0

    # Генерируем варианты изображения
    variants = generate_image_variants(image_array)
    
    best_plate = None
    best_confidence = 0.0
    logs = []

    # Пробуем распознать каждый вариант, пока не найдем идеальный номер
    for i, img_variant in enumerate(variants):
        try:
            # Запуск OCR
            result = ocr_reader.ocr(img_variant, cls=False)
            
            if not result or not result[0]:
                continue

            # Перебор всех найденных блоков текста
            for line in result[0]:
                raw_text = line[1][0]
                confidence = line[1][1]
                
                # 1. Нормализация (убираем мусор, кириллизация)
                normalized = normalize_chars(raw_text)
                
                # 2. Попытка исправить номер под формат L DDD LL
                fixed_plate = try_fix_plate(normalized)
                
                log_entry = f"Var{i}: '{raw_text}' -> '{normalized}' -> '{fixed_plate}' ({confidence:.2f})"
                logs.append(log_entry)

                if fixed_plate:
                    # Если нашли валидный номер, сохраняем его
                    if confidence > best_confidence:
                        best_plate = fixed_plate
                        best_confidence = confidence
            
            # Если на этом варианте изображения нашли номер с высокой уверенностью, выходим досрочно
            if best_plate and best_confidence > 0.90:
                break

        except Exception as e:
            print(f"Ошибка на варианте {i}: {e}")
            continue

    explanation = " | ".join(logs[-5:]) # Последние 5 попыток для лога
    return explanation, best_plate, best_confidence

# =================================================================
# FLASK ROUTES
# =================================================================

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json
    base64_img = data.get('image_data')
    
    if not base64_img:
        return jsonify({"error": "Нет данных"}), 400

    try:
        image_bytes = base64.b64decode(base64_img)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Ошибка декодирования")

        explanation, plate, conf = process_ocr_pipeline(img)

        if plate:
            res = {
                "plate": plate,
                "confidence_explanation": f"Найден: {plate} ({conf:.2f}). Лог: {explanation}"
            }
        else:
            res = {
                "plate": "",
                "confidence_explanation": f"Не найдено. Лог: {explanation}"
            }
        return jsonify(res), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK", "ready": ocr_reader is not None}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)