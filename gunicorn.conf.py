# -*- coding: utf-8 -*-
# Gunicorn: настройки под тяжёлый PaddleOCR в Docker (долгие запросы, один воркер).
bind = "0.0.0.0:5000"
workers = 1
worker_class = "sync"
threads = 1

# По умолчанию 30 с — воркер убивается при долгом OCR; для CPU это критично.
timeout = 300
graceful_timeout = 120
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

# Не перезапускать воркер «по кругу» — у Paddle тяжёлая инициализация
max_requests = 0
