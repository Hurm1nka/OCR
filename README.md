# OCR: распознавание российских госномеров

Веб-сервис на **Flask** + **PaddleOCR**: загрузка изображения, камера в браузере, IP-камера (URL кадра) и локальная камера на стороне сервера (OpenCV). Результат — номер в основном формате **БЦЦЦББ** (одна буква + три цифры + две буквы), дополнительно может возвращаться полный номер с регионом.

## Состав проекта

| Файл | Назначение |
|------|------------|
| `app.py` | Flask API, логика OCR и номера |
| `index.html` | Одностраничный интерфейс (Tailwind CDN) |
| `Dockerfile` | Образ Python 3.9 + зависимости + приложение |
| `gunicorn.conf.py` | Таймауты и воркер Gunicorn под тяжёлый OCR |
| `requirements.txt` | Зависимости для локального запуска без Docker |

## Быстрый старт (Docker)

В каталоге с `Dockerfile`:

```bash
docker build -t ocr-service:1 .
docker rm -f ocr-service 2>/dev/null
docker run -d --name ocr-service -p 3020:5000 ocr-service:1
```

- В контейнере приложение слушает порт **5000** (`gunicorn.conf.py`: `bind = "0.0.0.0:5000"`).
- Схема `-p 3020:5000`: **порт хоста 3020** → **порт контейнера 5000**.

Проверка:

```bash
curl -sS http://127.0.0.1:3020/health
```

Открыть UI: `http://127.0.0.1:3020/` (или через nginx — см. ниже).

### Если в образе «старая» версия кода

Docker кэширует слои. Полная пересборка:

```bash
docker build --no-cache -t ocr-service:1 .
```

Или инвалидация слоя с приложением (см. `ARG CACHEBUST` в `Dockerfile`):

```bash
docker build --build-arg CACHEBUST="$(date +%s)" -t ocr-service:1 .
```

Собирайте **из того же каталога**, где лежат актуальные `app.py` и `index.html`.

## Локальный запуск (без Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# При необходимости подберите версии paddlepaddle/paddleocr как в Dockerfile
python app.py
```

По умолчанию в `if __name__ == '__main__'` приложение слушает порт **3020** (см. конец `app.py`).

## HTTP API

Базовый URL — тот же, что у веб-страницы (или задан через `?api=` / meta, см. раздел про фронт).

### `GET /health`

Проверка готовности модели.

**Ответ 200 (JSON):**

- `status`: `"OK"`
- `ready` / `model_loaded`: `true`, если PaddleOCR инициализирован

### `GET /` и `GET /index.html`

Отдача `index.html` с заголовками против агрессивного кэша.

### `POST /process`

Распознавание по **base64** изображения (тело JSON).

**Тело:**

```json
{ "image_data": "<base64 без префикса data:image/...>" }
```

**Успех 200 (JSON, основные поля):**

| Поле | Описание |
|------|----------|
| `plate` | Основная часть номера (без региона), формат `БЦЦЦББ` |
| `plate_base` | 6 символов: буква + 3 цифры + 2 буквы |
| `plate_full` | Полный номер: основная часть + регион (если найден) |
| `region` | Регион: `""`, 2 или 3 цифры |
| `is_duplicate_recent` | `true`, если этот номер уже распознавался в последние `dedup_window_seconds` |
| `dedup_window_seconds` | Окно антидублирования в секундах (по умолчанию 30) |
| `confidence_explanation` | Краткий текст |
| `model_explanation` | Пояснение по шаблону / уверенности |
| `ocr_log` | Сжатый лог последних шагов пайплайна |

**Ошибки:** `400` (нет `image_data`), `500` (исключение при декодировании/OCR).

### `POST /process_ip_camera`

Сервер **сам** запрашивает картинку по URL (например `http://192.168.1.50:8080/shot.jpg`), затем тот же пайплайн OCR.

**Тело:**

```json
{ "camera_url": "http://..." }
```

**Успех 200:** те же поля, что у `/process`, плюс `preview_data_url` — `data:image/jpeg;base64,...` последнего загруженного кадра.

Таймаут чтения URL в коде — **6 секунд**.

### `POST /process_local_camera`

Кадр с **локальной** камеры машины, где крутится контейнер/процесс (OpenCV `VideoCapture`).

**Тело:**

```json
{ "camera_index": 0 }
```

**Успех 200:** как у IP-камеры, с `preview_data_url` при успешном JPEG-кодировании кадра.

### `POST /barrier/open`

Ручная команда открытия шлагбаума (пока заглушка).

**Тело (опционально):**

```json
{ "reason": "manual_ui" }
```

**Ответ 200 (JSON):**

- `ok`: `true`
- `message`: подтверждение, что заглушка вызвана
- `opened_at`: unix timestamp события

На **Windows** в коде используется `cv2.CAP_DSHOW` для индекса камеры.

## Маска номера (логика в `app.py`)

- **Основная часть (6 символов):** одна буква из набора госномера РФ **А В Е К М Н О Р С Т У Х**, затем **три цифры**, затем **две буквы** из того же набора. Проверка: `PLATE_REGEX`.
- **С регионом:** после основной части допускаются **0, 2 или 3** цифры региона. Проверка: `PLATE_WITH_REGION_REGEX`.
- Строка OCR нормализуется в канонический алфавит номера (`A B E K M H O P C T Y X`) и цифры, затем применяется коррекция частых путаниц (`DIGIT_TO_LETTER`, `LETTER_TO_DIGIT`).

## YOLO-детекция номера (добавлено)

Перед OCR можно использовать YOLO-детектор номера:

- backend ищет номер через YOLO и сначала запускает OCR по найденным crop;
- если YOLO не установлен или файл модели отсутствует, пайплайн автоматически работает в fallback-режиме (без YOLO).

Настройки через переменные окружения:

- `YOLO_MODEL_PATH` — путь к `.pt` модели (по умолчанию `models/license_plate.pt`);
- `YOLO_CONF_THRESHOLD` — порог детекции (по умолчанию `0.25`).

## Справочник функций `app.py`

### Инициализация и HTTP-хуки

| Имя | Назначение |
|-----|------------|
| `log_request` | `before_request`: логирует метод, путь, IP, `Content-Type`, размер тела |
| `log_response` | `after_request`: логирует статус ответа |
| `index_page` | `GET /`, `GET /index.html` — отдача `index.html` |
| `process_data` | `POST /process` — декодирование base64, вызов пайплайна, JSON-ответ |
| `process_ip_camera` | `POST /process_ip_camera` — загрузка кадра по URL, OCR, превью base64 |
| `process_local_camera` | `POST /process_local_camera` — кадр с локальной камеры, OCR |
| `health` | `GET /health` — статус модели |

### Нормализация и извлечение номера из текста

| Имя | Назначение |
|-----|------------|
| `normalize_chars` | Верхний регистр, латиница→кириллица по таблице, отбор символов, допустимых на номере |
| `try_fix_plate` | Скользящее окно 6 символов, подстановки в позициях букв/цифр, возврат строки только если совпала `PLATE_REGEX` |
| `try_extract_plate_from_text` | Поиск основной части через `try_fix_plate`, разбор хвоста как региона (0/2/3 цифры), проверка `PLATE_WITH_REGION_REGEX` |

### Обработка изображения

| Имя | Назначение |
|-----|------------|
| `upscale` | Увеличение изображения ×2 (bicubic) |
| `deskew_image` | Выравнивание наклона по маске содержимого (minAreaRect) |
| `generate_image_variants` | Цепочка: deskew → upscale, затем варианты: исходное цветное, CLAHE, OTSU, adaptive, adaptive inverted |
| `decode_image_from_bytes` | `imdecode` из байтов JPEG/PNG и т.п. |

### Пайплайн OCR

| Имя | Назначение |
|-----|------------|
| `process_ocr_pipeline` | Для каждого варианта изображения вызывает PaddleOCR, фильтрует боксы по соотношению сторон, извлекает номер; при уверенности > 0.9 может выйти раньше; fallback — объединение всех строк OCR на исходном кадре |

**Возвращает кортеж:** `(model_explanation, ocr_log, best_plate, best_region, best_confidence)`.

## Фронтенд (`index.html`)

### Вычисление URL API

1. Параметр **`?api=https://хост/путь`** (без завершающего `/` у базы — обрежется).
2. Иначе **`<meta name="ocr-api-base" content="...">`**.
3. Иначе при **`file://`** — база `http://127.0.0.1:3020`.
4. Иначе — относительно **каталога текущей страницы** (`pageDirectoryBaseHref`), чтобы при публикации в подкаталоге (например `/ocr/`) запросы шли на `/ocr/health`, `/ocr/process`, и т.д.

От этого базируются:

- `HEALTH_CHECK_URL`, `OCR_SERVICE_URL`, `IP_CAMERA_PROCESS_URL`, `LOCAL_CAMERA_PROCESS_URL`.

### Функции в `<script>`

| Имя | Назначение |
|-----|------------|
| `pageDirectoryBaseHref` | URL каталога страницы (для относительных API в подпути) |
| `displayError` / `clearError` | Блок ошибки внизу страницы |
| `setLoadingState` | Спиннер и блокировка кнопок (захват, файл, IP, локальная камера) |
| `checkServiceStatus` | `GET` health, обновление баннера статуса |
| `processImage` | `POST /process` с `{ image_data }` |
| `startVideoStream` | Вкл/выкл `getUserMedia`, предпочтение задней камеры |
| `captureFrame` | Кадр с `<video>` в canvas → JPEG base64 → `processImage` |
| `processIpCameraFrame` | `POST /process_ip_camera` с `{ camera_url }` |
| `processLocalCameraFrame` | `POST /process_local_camera` с `{ camera_index }` |
| `tickIpLive` | Один цикл живого режима IP (с защитой от наложения запросов) |
| `stopIpLiveMode` / `startStopIpLiveMode` | Интервальный опрос IP-камеры |

Периодический опрос **`/health` каждые 5 с** после загрузки страницы.

## Nginx и подкаталог (пример)

Если UI открыт как `https://example.com/ocr/`, прокси обычно делают с **срезом префикса** на бэкенд:

```nginx
location /ocr/ {
    proxy_pass http://127.0.0.1:3020/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Тогда браузер вызывает `https://example.com/ocr/health`, nginx передаёт на контейнер `GET /health`.

## Gunicorn (`gunicorn.conf.py`)

| Параметр | Значение | Смысл |
|----------|----------|--------|
| `bind` | `0.0.0.0:5000` | Порт внутри контейнера |
| `workers` | `1` | Один процесс — модель в памяти один раз |
| `timeout` | `300` | Долгий CPU-OCR не обрывается по умолчанию |
| `accesslog` / `errorlog` | `-` | Логи в stdout/stderr |

## Переменные окружения в образе (`Dockerfile`)

Ограничение потоков BLAS/OpenMP (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MALLOC_ARENA_MAX`) и `PYTHONUNBUFFERED=1` для предсказуемых логов.

## Безопасность (кратко)

- **`/process_ip_camera`** заставляет **сервер** ходить по произвольному URL — используйте только в доверенной сети или закройте endpoint на уровне nginx/файрвола.
- CORS включён глобально (`flask_cors`) — для публичного интернета лучше сузить политику.

## Лицензии сторонних компонентов

Модели **PaddleOCR** подгружаются при первом запуске с CDN Baidu; убедитесь, что это допустимо в вашей среде.
