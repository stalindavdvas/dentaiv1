import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import pandas as pd

# Configuración de rutas
DATASET_DIR = "C:/datasets/dentiset"
MODEL_DIR = os.path.join(DATASET_DIR, "modelos")
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuración del modelo
IMG_SIZE = (512, 512)  # Tamaño de entrada ajustado
BATCH_SIZE = 16
EPOCHS = 100
CLASSES = ["cancer", "caries", "gingivitis", "perdidos", "ulceras"]
NUM_CLASSES = len(CLASSES)

# Mapeo de clases a índices
label_map = {cls: idx for idx, cls in enumerate(CLASSES)}

# Función para cargar anotaciones
def load_annotations(dataset_dir, classes):
    image_data = {}
    for class_name in classes:
        annotations_file = os.path.join(dataset_dir, "train", class_name, "_annotations.csv")
        if not os.path.exists(annotations_file):
            print(f"Archivo de anotaciones no encontrado: {annotations_file}")
            continue

        annotations = pd.read_csv(annotations_file)
        for _, row in annotations.iterrows():
            filename = os.path.join("train", class_name, row['filename'])
            label = class_name
            box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            width = row['width']
            height = row['height']

            normalized_box = normalize_boxes([box], width, height)[0]

            if filename not in image_data:
                image_data[filename] = {'boxes': [], 'labels': []}
            image_data[filename]['boxes'].append(normalized_box)
            image_data[filename]['labels'].append(label)

    return image_data

# Normalización de cajas delimitadoras
def normalize_boxes(boxes, width, height):
    return np.array([
        [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
        for box in boxes
    ], dtype=np.float32)

# Preprocesamiento de imágenes
def preprocess_image(image_path, target_size=(512, 512)):
    original_img = cv2.imread(image_path)
    h, w, _ = original_img.shape

    # Mantener proporciones al redimensionar
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_img = cv2.resize(original_img, (new_w, new_h))
    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_img = padded_img / 255.0  # Normalizar
    return padded_img

# Generador de datos con aumentación
class DataGenerator(Sequence):
    def __init__(self, image_annotations, dataset_dir, batch_size, target_size):
        self.image_annotations = image_annotations
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.filenames = list(image_annotations.keys())
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def __getitem__(self, index):
        batch_filenames = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y_class, Y_bbox = [], [], []

        for filename in batch_filenames:
            img_path = os.path.join(self.dataset_dir, filename)
            if not os.path.exists(img_path):
                continue

            img = preprocess_image(img_path, target_size=self.target_size)
            boxes = self.image_annotations[filename]['boxes']
            labels = self.image_annotations[filename]['labels']

            X.append(img)
            Y_class.append(label_map[labels[0]])
            Y_bbox.append(boxes[0])

        X = np.array(X, dtype=np.float32)
        Y_class = np.array(Y_class, dtype=np.int32)
        Y_bbox = np.array(Y_bbox, dtype=np.float32)

        for i in range(len(X)):
            X[i] = self.datagen.random_transform(X[i])

        return X, {"class_output": Y_class, "bbox_output": Y_bbox}

# Crear modelo personalizado
def create_custom_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Bloques convolucionales
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Aplanado
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    # Salidas
    class_output = layers.Dense(num_classes, activation='softmax', name="class_output")(x)
    bbox_output = layers.Dense(4, activation='sigmoid', name="bbox_output")(x)

    return Model(inputs=inputs, outputs=[class_output, bbox_output])

# Métrica IoU personalizada
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

# Preparar datos
image_annotations = load_annotations(DATASET_DIR, CLASSES)
train_gen = DataGenerator(image_annotations, DATASET_DIR, BATCH_SIZE, IMG_SIZE)

# Crear modelo
model = create_custom_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES)

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        "class_output": "sparse_categorical_crossentropy",
        "bbox_output": "mean_squared_error"
    },
    metrics={
        "class_output": "accuracy",
        "bbox_output": iou_metric
    }
)

# Entrenar modelo
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    ]
)

# Guardar modelo
model_path = os.path.join(MODEL_DIR, "modelo_dental.h5")
model.save(model_path)
print(f"Modelo guardado en: {model_path}")
