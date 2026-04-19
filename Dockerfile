# Используем Python версии 3.9
FROM python:3.9-slim

# Установка системных зависимостей для OpenCV и build-tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# === КРИТИЧЕСКИЙ ШАГ: ВСЕ ЗАВИСИМОСТИ В ОДНОЙ КОМАНДЕ ===
# Это решает проблему конфликта ABI. numpy==1.22.4 - известная стабильная версия для Python 3.9. Flask показ
RUN pip install --no-cache-dir \
    numpy==1.22.4 \
    paddlepaddle==2.6.2 \
    paddleocr==2.6.1.3 \
    opencv-python \
    flask \
    flask-cors \
    gunicorn \
    ultralytics
    

# Устанавливаем рабочую директорию
WORKDIR /app

# Старый код в образе чаще всего из-за кэша слоёв Docker.
# Полная пересборка: docker build --no-cache -t ocr-service:1 .
# Только обновить слой с приложением (без переустановки pip): см. CACHEBUST ниже.
ARG CACHEBUST=0
RUN echo "build=${CACHEBUST}"

# Копируем файлы приложения (этот слой пересобирается после смены CACHEBUST или после правок файлов)
COPY app.py database.py index.html gunicorn.conf.py ./

# Ограничение потоков BLAS/OpenMP снижает риск OOM и «штормов» CPU в контейнере
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1

# Открываем порт для flask (бэкенд)
EXPOSE 5000

# Gunicorn: см. gunicorn.conf.py (таймауты для OCR, один воркер)
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]