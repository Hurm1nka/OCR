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
    gunicorn
    

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы приложения
COPY app.py .
COPY index.html .

# Открываем порт для flask (бэкенд)
EXPOSE 5000

# Команда для запуска веб-сервера Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]