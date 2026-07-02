import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
import os
import json
import time
from datetime import datetime

# ----------------------------- Конфигурация -----------------------------
MODELS_PATH = "models"
HISTORY_FILE = "history.json"
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tiff"]

# ----------------------------- Загрузка моделей -----------------------------
@st.cache_resource
def load_yolo_model(model_name):
    model_path = os.path.join(MODELS_PATH, model_name)
    if not os.path.exists(model_path):
        return None
    try:
        return YOLO(model_path)
    except:
        return None

@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'], gpu=False)  # CPU для стабильности

# ----------------------------- Обработка изображения -----------------------------
def process_image(image, model, reader):
    start = time.time()
    results = model(image, verbose=False)
    boxes = []
    confs = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            confs.append(float(box.conf[0]))
    
    plates = []
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        text_list = reader.readtext(crop, detail=0)
        text = text_list[0] if text_list else ""
        plates.append({
            "bbox": (x1, y1, x2, y2),
            "detection_conf": conf,
            "text": text
        })
    elapsed = time.time() - start
    return plates, elapsed

def draw_annotations(image, plates):
    img = image.copy()
    for p in plates:
        x1, y1, x2, y2 = p["bbox"]
        text = p["text"]
        color = (0, 255, 0) if text else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if text:
            cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

# ----------------------------- Работа с историей -----------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def add_record(history, filename, model_name, plates, elapsed):
    record = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "model": model_name,
        "num_plates": len(plates),
        "num_recognized": sum(1 for p in plates if p["text"]),
        "plates": [{"text": p["text"], "conf": p["detection_conf"]} for p in plates],
        "elapsed": round(elapsed, 2)
    }
    history.append(record)
    save_history(history)
    return record

# ----------------------------- Интерфейс -----------------------------
st.set_page_config(page_title="ALPR - Распознавание номеров", layout="wide")
st.title("🚗 Распознавание номерных знаков")

# Боковая панель
with st.sidebar:
    st.header("Настройки")
    # Список моделей
    model_files = [f for f in os.listdir(MODELS_PATH) if f.endswith('.pt')]
    if not model_files:
        st.error("❌ Нет моделей в папке models/")
        st.stop()
    selected_model = st.selectbox("Выберите модель YOLO", model_files)
    
    st.header("Загрузка")
    uploaded = st.file_uploader("Выберите изображение", type=ALLOWED_EXTENSIONS)
    
    # Кнопка обработки
    process_btn = st.button("🔍 Распознать", type="primary", use_container_width=True)

# Основная область
tab1, tab2, tab3 = st.tabs(["📸 Распознавание", "📜 История", "📊 Статистика"])

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        if uploaded is not None:
            image = Image.open(uploaded)
            image_np = np.array(image.convert('RGB'))
            st.image(image_np, caption="Исходное изображение", use_container_width=True)
        else:
            image_np = None
            st.info("Загрузите изображение через боковую панель.")
    
    with col_right:
        if process_btn and uploaded is not None:
            with st.spinner("Обработка..."):
                model = load_yolo_model(selected_model)
                if model is None:
                    st.error("Не удалось загрузить модель.")
                else:
                    reader = load_easyocr()
                    plates, elapsed = process_image(image_np, model, reader)
                    
                    # Сохраняем историю
                    history = load_history()
                    add_record(history, uploaded.name, selected_model, plates, elapsed)
                    
                    st.success(f"✅ Обработано за {elapsed:.2f} сек. Найдено {len(plates)} номеров.")
                    
                    # Аннотированное изображение
                    annotated = draw_annotations(image_np, plates)
                    st.image(annotated, caption="Результат", use_container_width=True)
                    
                    # Детали
                    if plates:
                        for i, p in enumerate(plates):
                            st.write(f"**Номер {i+1}**")
                            st.write(f"- Координаты: {p['bbox']}")
                            st.write(f"- Уверенность: {p['detection_conf']:.2f}")
                            st.write(f"- Текст: **{p['text'] if p['text'] else 'не распознан'}**")
                            st.divider()
                    else:
                        st.warning("Номерных знаков не обнаружено.")
        elif process_btn and uploaded is None:
            st.warning("Пожалуйста, загрузите изображение.")

with tab2:
    st.header("История распознаваний")
    history = load_history()
    if not history:
        st.info("История пуста.")
    else:
        # Простая таблица
        for idx, rec in enumerate(reversed(history)):
            st.write(f"**{idx+1}. {rec['timestamp']}**")
            st.write(f"Файл: {rec['filename']} | Модель: {rec['model']}")
            st.write(f"Найдено: {rec['num_plates']}, распознано: {rec['num_recognized']}, время: {rec['elapsed']} сек.")
            if rec['plates']:
                st.write("Номера:", ", ".join([p['text'] for p in rec['plates'] if p['text']]))
            st.divider()
        
        if st.button("Очистить историю"):
            save_history([])
            st.rerun()

with tab3:
    st.header("Статистика")
    history = load_history()
    if not history:
        st.info("Нет данных.")
    else:
        total_images = len(history)
        total_plates = sum(rec['num_plates'] for rec in history)
        total_recognized = sum(rec['num_recognized'] for rec in history)
        avg_time = sum(rec['elapsed'] for rec in history) / total_images if total_images else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего изображений", total_images)
        col2.metric("Найдено номеров", total_plates)
        col3.metric("Распознано номеров", total_recognized)
        col4.metric("Среднее время", f"{avg_time:.2f} сек")
        
        # Список распознанных номеров
        all_texts = []
        for rec in history:
            for p in rec['plates']:
                if p['text']:
                    all_texts.append(p['text'])
        if all_texts:
            st.write("**Распознанные номера:**")
            st.write(", ".join(set(all_texts)))