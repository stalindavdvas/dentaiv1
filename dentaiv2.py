from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import custom_object_scope
from ultralytics import YOLO  # Importar YOLO desde ultralytics
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import PosixPath, WindowsPath
import pathlib
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd  # Para leer el archivo CSV
import torch

# Configuración inicial
pathlib.PosixPath = pathlib.WindowsPath

# Cargar BioBERT para preguntas y respuestas
model_name = "dmis-lab/biobert-base-cased-v1.2"  # Modelo BioBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Inicializar la aplicación Flask y SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Configurar directorios fijos
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
ORIGINAL_IMAGE = os.path.join(UPLOAD_FOLDER, 'original.jpg')
YOLO_RESULT_IMAGE = os.path.join(RESULTS_FOLDER, 'yolo_result.jpg')
CUSTOM_RESULT_IMAGE = os.path.join(RESULTS_FOLDER, 'custom_result.jpg')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Cargar modelo YOLOv8
yolo_model = YOLO('modelos/bestyolo8.pt')  # Cambia 'yolo8n.pt' por la ruta de tu modelo YOLOv8

# Cargar modelo personalizado
label_map = {0: "cancer", 1: "caries", 2: "gingivitis", 3: "perdidos", 4: "ulceras"}

def iou_metric(y_true, y_pred):
    x1_true, y1_true, x2_true, y2_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, x2_pred, y2_pred = tf.split(y_pred, 4, axis=-1)
    xi1 = tf.maximum(x1_true, x1_pred)
    yi1 = tf.maximum(y1_true, y1_pred)
    xi2 = tf.minimum(x2_true, x2_pred)
    yi2 = tf.minimum(y2_true, y2_pred)
    inter_area = tf.maximum(xi2 - xi1, 0) * tf.maximum(yi2 - yi1, 0)
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union_area = true_area + pred_area - inter_area
    return tf.reduce_mean(inter_area / union_area)

with custom_object_scope({'iou_metric': iou_metric}):
    custom_model = load_model('modelos/modelo_dental.h5')

# Función para preprocesar imágenes
def preprocess_image(image_path):
    original_img = cv2.imread(image_path)
    img_resized = cv2.resize(original_img, (512, 512))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return original_img, img_array

# Función para predecir con el modelo personalizado
def predict_custom_model(image_path):
    original_img, img_array = preprocess_image(image_path)
    class_output, bbox_output = custom_model.predict(img_array)
    predicted_class = np.argmax(class_output, axis=1)[0]
    class_name = label_map[predicted_class]
    confidence = class_output[0][predicted_class]
    bbox = bbox_output[0]
    height, width, _ = original_img.shape
    x1, y1, x2, y2 = (bbox * [width, height, width, height]).astype(int)
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(original_img, (x1, y1), (x2, y2), color, thickness)
    text = f"{class_name}: {confidence * 100:.1f}%"
    cv2.putText(original_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(CUSTOM_RESULT_IMAGE, original_img)
    return class_name, confidence

# Función para procesar con YOLO
def process_yolo(image_path):
    original_img = cv2.imread(image_path)
    results = yolo_model(image_path)  # YOLOv8 devuelve resultados directamente
    detected_diseases = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Obtener las cajas delimitadoras
        confidences = result.boxes.conf.cpu().numpy()  # Obtener las confianzas
        class_ids = result.boxes.cls.cpu().numpy()  # Obtener los IDs de las clases
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = yolo_model.names[int(cls_id)]  # Obtener el nombre de la clase
            confidence = float(conf)
            color = (0, 255, 0)
            cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
            label_text = f'{label} {confidence:.2f}'
            cv2.putText(original_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detected_diseases.append((label, confidence))
    cv2.imwrite(YOLO_RESULT_IMAGE, original_img)
    return detected_diseases

# Cargar el archivo CSV de recomendaciones
recommendations_df = pd.read_csv('recomendacion_corregido.csv')
recommendations_df['confianza'] = recommendations_df['confianza'].str.rstrip('%').astype(float)

# Función para obtener una recomendación inicial desde el CSV
def get_initial_recommendation(disease, confidence):
    filtered_df = recommendations_df[recommendations_df['enfermedad'] == disease]
    if filtered_df.empty:
        return "No se encontraron recomendaciones específicas para esta enfermedad."
    closest_row = filtered_df[filtered_df['confianza'] <= confidence].sort_values(by='confianza', ascending=False).iloc[0]
    return closest_row['recomendación']

# Función para refinar la recomendación usando BioBERT
def refine_recommendation(initial_recommendation, disease, confidence):
    # Crear un contexto basado en la recomendación inicial y el archivo CSV
    context = recommendations_df[recommendations_df['enfermedad'] == disease]['recomendación'].str.cat(sep=" ")
    question = (
        f"¿Cuál es la mejor recomendación médica para {disease} con una confianza de {confidence}%? "
        f"Considera la siguiente información: {initial_recommendation}"
    )
    try:
        # Tokenizar la pregunta y el contexto
        inputs = tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        return answer.strip()
    except Exception as e:
        print(f"Error al generar la recomendación: {e}")
        return "Ocurrió un error al generar la recomendación. Consulta a un especialista."

@app.route('/')
def index():
    return render_template('inicio4.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'})
    file.save(ORIGINAL_IMAGE)
    # Predicciones del modelo personalizado
    custom_class, custom_conf = predict_custom_model(ORIGINAL_IMAGE)
    # Predicciones del modelo YOLO
    yolo_detections = process_yolo(ORIGINAL_IMAGE)
    yolo_results = [
        {'disease': disease, 'confidence': f"{confidence * 100:.1f}%"}
        for disease, confidence in yolo_detections
    ]
    # Seleccionar la mejor detección
    all_detections = [(custom_class, custom_conf)] + [(disease, confidence) for disease, confidence in yolo_detections]
    best_detection = max(all_detections, key=lambda x: x[1])  # Selecciona la detección con la mayor confianza
    best_disease, best_confidence = best_detection
    # Obtener la recomendación inicial desde el archivo CSV
    initial_recommendation = get_initial_recommendation(best_disease, best_confidence * 100)  # Convertir a porcentaje
    # Refinar la recomendación usando BioBERT
    refined_recommendation = refine_recommendation(initial_recommendation, best_disease, best_confidence * 100)
    return jsonify({
        'original': '/static/uploads/original.jpg',
        'custom_result': '/static/results/custom_result.jpg',
        'yolo_result': '/static/results/yolo_result.jpg',
        'custom_detections': {
            'disease': custom_class,
            'confidence': f"{custom_conf * 100:.1f}%"
        },
        'yolo_detections': yolo_results,
        'recommendation': refined_recommendation  # Aquí devolvemos la recomendación refinada
    })

@socketio.on('message')
def handle_message(message):
    timestamp = datetime.now().strftime('%H:%M')
    emit('message', {'msg': message, 'timestamp': timestamp}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='127.0.0.1', port=5001)