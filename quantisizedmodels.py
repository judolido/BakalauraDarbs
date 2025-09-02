import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib
from tensorflow.keras.callbacks import ReduceLROnPlateau

# === Settings ===
DATASET_PATH = "H:/Dataset"
CLASS_MAP = {"N": 0, "B": 1, "OR": 2}
#SAMPLES_PER_FILE = 16384
SAMPLES_PER_FILE = 96000
def load_all_data(base_path, class_map, max_files_per_class=10, chunk_size=16384, step_size=8192):
    all_data = []  # Will hold (chunk, label) tuples

    for class_folder, label in class_map.items():
        class_path = os.path.join(base_path, class_folder)
        file_list = sorted([f for f in os.listdir(class_path) if f.endswith(".csv")])[:max_files_per_class]

        for file_name in tqdm(file_list, desc=class_folder):
            file_path = os.path.join(class_path, file_name)
            try:
                df = pd.read_csv(file_path, header=None, comment='t')
                df = df.apply(pd.to_numeric, errors='coerce')
                signal = df.dropna().values.flatten().astype(np.float32)

                signal = signal.astype(np.float32)

                # Slide over the signal in steps
                for start in range(0, len(signal) - chunk_size + 1, step_size):
                    chunk = signal[start:start + chunk_size]

                    # Compute FFT
                    fft_chunk = np.abs(np.fft.rfft(chunk))
                    fft_chunk = fft_chunk[:chunk_size // 2]

                    # Normalize
                    fft_chunk = (fft_chunk - np.mean(fft_chunk)) / (np.std(fft_chunk) + 1e-8)

                    all_data.append((fft_chunk, label))

            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    print(f"Total chunks collected: {len(all_data)}")

    # Shuffle and split
    np.random.shuffle(all_data)
    X, y = zip(*all_data)
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


X, y = load_all_data(DATASET_PATH, CLASS_MAP)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def representative_data_gen():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

# === Compact 1D CNN model ===
def build_deep_cnn_classifier():
    input_layer = keras.Input(shape=(4096*2, 1))
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(input_layer)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs=input_layer, outputs=output)


# === Train, evaluate and export ===
def train_and_export():
    model_name = "Small_LSTM"
    output_dir = f"results_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    model = build_deep_cnn_classifier()
    model.compile(optimizer=keras.optimizers.Adam(0.005),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=20,
                        callbacks=[ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, verbose=1)],
                        verbose=1)

    # Save training loss plot
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Predict
    y_pred = np.argmax(model.predict(X_val), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "IR", "OR"])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Save classification report
    report = classification_report(y_val, y_pred, target_names=["N", "IR", "OR"])
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Save .h5 model
    model.save(os.path.join(output_dir, "model.h5"))

    # Save float32 TFLite
    tflite_path = os.path.join(output_dir, "model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Save quantized int8 TFLite
    int8_path = os.path.join(output_dir, "model_int8.tflite")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tf.int8
    #converter.inference_output_type = tf.int8
    int8_model = converter.convert()
    with open(int8_path, "wb") as f:
        f.write(int8_model)

train_and_export()
