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
import google.generativeai as genai  # Para Gemini
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

# Configuración inicial
pathlib.PosixPath = pathlib.WindowsPath

# Configurar Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "No se encontró la clave API de Gemini. Asegúrate de configurar la variable de entorno GEMINI_API_KEY.")
genai.configure(api_key=API_KEY)

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


# Función para determinar si la imagen es a color o en escala de grises
def is_grayscale(image_path):
    img = cv2.imread(image_path)
    if len(img.shape) == 2:  # Imagen en escala de grises
        return True
    elif len(img.shape) == 3 and img.shape[2] == 1:  # Imagen con un solo canal
        return True
    else:  # Imagen a color (3 canales)
        return False


# Función para cargar el modelo adecuado según el tipo de imagen
def load_model_based_on_image(image_path):
    if is_grayscale(image_path):
        print("Imagen detectada como rayos X. Cargando modelo 'bestrayosx.pt'...")
        return YOLO('modelos/bestrayosx.pt')  # Modelo para rayos X
    else:
        print("Imagen detectada como a color. Cargando modelo 'bestyolo8.pt'...")
        return YOLO('modelos/best.pt')  # Modelo para imágenes a color


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
    custom_model = load_model('modelos/modelo_dentalv1.h5')


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


# Función para procesar con el modelo YOLO seleccionado
def process_yolo_with_selected_model(image_path, selected_model):
    original_img = cv2.imread(image_path)
    results = selected_model(image_path)  # YOLOv8 devuelve resultados directamente
    detected_diseases = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Obtener las cajas delimitadoras
        confidences = result.boxes.conf.cpu().numpy()  # Obtener las confianzas
        class_ids = result.boxes.cls.cpu().numpy()  # Obtener los IDs de las clases
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = selected_model.names[int(cls_id)]  # Obtener el nombre de la clase
            confidence = float(conf)
            color = (0, 255, 0)
            cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
            label_text = f'{label} {confidence:.2f}'
            cv2.putText(original_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detected_diseases.append((label, confidence))
    cv2.imwrite(YOLO_RESULT_IMAGE, original_img)
    return detected_diseases


# Función para generar embeddings con Gemini
# Función para generar embeddings con Gemini
def generate_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",  # Modelo de embeddings
            content=text
        )
        return result['embedding']
    except Exception as e:
        print(f"Error al generar el embedding: {e}")
        return None


# Diccionario de enfermedades y sus descripciones
diseases = {
    "caries": "Enfermedad dental causada por bacterias que destruyen el esmalte.",
    "gingivitis": "Inflamación de las encías debido a la acumulación de placa.",
    "cancer": "Crecimiento anormal de células en la cavidad oral.",
    "ulceras": "Llagas o heridas en la mucosa oral.",
    "perdidos": "Pérdida de dientes debido a trauma o enfermedad."
}

# Generar y almacenar embeddings
disease_embeddings = {disease: generate_embedding(description) for disease, description in diseases.items()}


# Función para calcular la similitud del coseno
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Función para encontrar la enfermedad más similar
def find_most_similar_disease(detected_disease, disease_embeddings):
    detected_embedding = generate_embedding(detected_disease)
    similarities = {
        disease: cosine_similarity(detected_embedding, embedding)
        for disease, embedding in disease_embeddings.items()
    }
    return max(similarities, key=similarities.get)


# Función para generar una recomendación con Gemini
def generate_recommendation_with_gemini(disease):
    print(disease)
    prompt = (
        f"Genera una recomendación médica detallada para {disease}, Habla como si fueras un odontologo profesional todo sobre la enfermedad detectada, todas las enfermedades que salen son de odontologia, "
        f"recomienda tratamientos caseros "
        f"Empieza diciendo: No reemplazo a un profesional en el área, pero mi recomendación es:"
    )
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error al generar la recomendación con Gemini: {e}")
        return "Ocurrió un error al generar la recomendación. Consulta a un especialista."


@app.route('/')
def index():
    return render_template('iniciodentaiv1.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'})
    file.save(ORIGINAL_IMAGE)

    # Seleccionar el modelo adecuado según el tipo de imagen
    selected_model = load_model_based_on_image(ORIGINAL_IMAGE)

    # Predicciones del modelo personalizado
    custom_class, custom_conf = predict_custom_model(ORIGINAL_IMAGE)

    # Predicciones del modelo YOLO seleccionado
    yolo_detections = process_yolo_with_selected_model(ORIGINAL_IMAGE, selected_model)
    yolo_results = [
        {'disease': disease, 'confidence': f"{confidence * 100:.1f}%"}
        for disease, confidence in yolo_detections
    ]

    # Seleccionar la mejor detección
    all_detections = [(custom_class, custom_conf)] + [(disease, confidence) for disease, confidence in yolo_detections]
    best_detection = max(all_detections, key=lambda x: x[1])  # Selecciona la detección con la mayor confianza
    best_disease, best_confidence = best_detection

    # Encontrar la enfermedad más similar usando embeddings
    most_similar_disease = find_most_similar_disease(best_disease, disease_embeddings)
    print(most_similar_disease)
    # Generar la recomendación con Gemini
    refined_recommendation = generate_recommendation_with_gemini(most_similar_disease)

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
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)