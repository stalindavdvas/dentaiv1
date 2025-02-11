import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import json

# ------------------------------
# Paso 1: Preparar el Dataset
# ------------------------------

# Cargar el CSV
csv_path = "recomendacion_corregido.csv"
data = pd.read_csv(csv_path)

# Crear prompts y targets
data['prompt'] = data.apply(lambda row: f"Genera una recomendación médica para el paciente que tiene la enfermedad {row['enfermedad']} con una confianza de {row['confianza']}. ¿Qué acción debe tomar? en español", axis=1)
data['target'] = data['recomendación']

# Guardar los prompts y targets en un nuevo archivo (opcional)
data[['prompt', 'target']].to_csv("dataset_preparado.csv", index=False)

# ------------------------------
# Paso 2: Tokenizar el Dataset
# ------------------------------

# Inicializar el tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(data['prompt'].tolist() + data['target'].tolist())

# Tokenizar prompts y targets
prompt_sequences = tokenizer.texts_to_sequences(data['prompt'])
target_sequences = tokenizer.texts_to_sequences(data['target'])

# Padding para asegurar que todas las secuencias tengan la misma longitud
max_sequence_length = max(max(len(seq) for seq in prompt_sequences), max(len(seq) for seq in target_sequences))
prompt_sequences = pad_sequences(prompt_sequences, maxlen=max_sequence_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')

# Guardar el tokenizer
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("Tokenizer guardado correctamente.")

# ------------------------------
# Paso 3: Construir el Modelo
# ------------------------------

# Parámetros del modelo
vocab_size = len(tokenizer.word_index) + 1  # Tamaño del vocabulario
embedding_dim = 128  # Dimensión del embedding
lstm_units = 256  # Unidades en la capa LSTM
learning_rate = 0.001

# Construir el modelo
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(lstm_units, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy')

# Resumen del modelo
model.summary()

# ------------------------------
# Paso 4: Entrenar el Modelo
# ------------------------------

# Preparar los datos de entrenamiento
X_train = np.array(prompt_sequences)
y_train = np.array(target_sequences)

# Entrenar el modelo
batch_size = 32
epochs = 50

history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

# Guardar el modelo entrenado
model.save("custom_model/model.h5")
print("Modelo guardado correctamente.")

# ------------------------------
# Paso 5: Probar el Modelo
# ------------------------------

def generate_recommendation(prompt, max_length=128, temperature=0.7):
    input_sequence = tokenizer.texts_to_sequences([prompt])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')

    generated_text = []
    current_sequence = input_sequence[0]

    for _ in range(max_length):
        predictions = model.predict(np.array([current_sequence]))[0][-1]
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        preds = exp_preds / np.sum(exp_preds)
        next_token = np.random.choice(len(preds), p=preds)

        if next_token == tokenizer.word_index.get("<EOS>"):
            break

        generated_text.append(next_token)
        current_sequence = np.append(current_sequence[1:], next_token)

    response = tokenizer.sequences_to_texts([generated_text])[0]
    return response.strip()

# Prueba el modelo
test_prompt = "Genera una recomendación médica para el paciente que tiene la enfermedad caries con una confianza de 90%. ¿Qué acción debe tomar? en español"
recommendation = generate_recommendation(test_prompt)
print("Recomendación:", recommendation)