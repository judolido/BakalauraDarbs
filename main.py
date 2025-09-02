import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ReduceLROnPlateau

# === Settings ===
DATASET_PATH = "H:/Dataset"
CLASS_MAP = {"N": 0, "B": 1, "OR": 2}
#SAMPLES_PER_FILE = 16384
SAMPLES_PER_FILE = 96000


def load_all_data(base_path, class_map, max_files_per_class=150, chunk_size=16384, step_size=8192):
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


# === Models ===
def build_cnn():
    return keras.Sequential([
        layers.Conv1D(32, 16, activation='relu', input_shape=(4096*2, 1)),
        layers.MaxPool1D(4),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

def build_mlp():
    return keras.Sequential([
        layers.Flatten(input_shape=(4096*2, 1)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])

def build_lstm():
    return keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(4096*2, 1)),
        layers.LSTM(32),
        layers.Dense(3, activation='softmax')
    ])

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

models = {
    "dCNN": build_deep_cnn_classifier,
    "MLP": build_mlp,
    "CNN": build_cnn,
    "LSTM": build_lstm

}

# === Training and Evaluation ===
import os
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ReduceLROnPlateau

def train_and_plot(model_name, model_builder):
    histories = []
    output_dir = f"results_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"classification_report_{model_name}.txt")

    with open(report_path, "w") as f_report:
        for run in range(2):

            print(f"\n🔁 Training {model_name} - Run {run+1}")
            model = model_builder()
            model.compile(
                keras.optimizers.Adam(learning_rate=0.005),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            #model_save_path = os.path.join(output_dir, f"{model_name}_run{run + 1}.h5")
            #model.save(model_save_path)
            #print(f"💾 Model saved to {model_save_path}")

            # Adaptive learning rate
            lr_callback = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-6
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=20,
                verbose=1,
                callbacks=[lr_callback]
            )
            histories.append(history)

            # === Predict & Evaluate ===
            y_pred = np.argmax(model.predict(X_val), axis=1)

            # Confusion Matrix
            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "IR", "OR"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - {model_name} Run {run+1}")
            cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}_run{run+1}.png")
            plt.savefig(cm_path)
            plt.close()

            # Classification Report
            report = classification_report(y_val, y_pred, target_names=["N", "IR", "OR"])
            print(f"📊 Classification Report - {model_name} Run {run+1}")
            print(report)
            f_report.write(f"Run {run+1}\n{report}\n{'='*50}\n")

            # Save the model in the same folder
            model_save_path = os.path.join(output_dir, f"{model_name}_run{run + 1}.h5")
            model.save(model_save_path)
            print(f"💾 Model saved to {model_save_path}")
            # Save standard TFLite model
            tflite_model_path = os.path.join(output_dir, f"{model_name}_run{run + 1}.tflite")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(tflite_model_path, "wb") as f:
                f.write(tflite_model)
            print(f"📦 TFLite model saved to {tflite_model_path}")

            # Optional: Save quantized TFLite model
            quant_model_path = os.path.join(output_dir, f"{model_name}_run{run + 1}_quant.tflite")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quant_tflite_model = converter.convert()
            with open(quant_model_path, "wb") as f:
                f.write(quant_tflite_model)
            print(f"📦 Quantized TFLite model saved to {quant_model_path}")

    # === Plot Loss Curves ===
    plt.figure()
    for i, h in enumerate(histories):
        plt.plot(h.history['loss'], label=f"Train Run {i+1}")
        plt.plot(h.history['val_loss'], label=f"Val Run {i+1}")
    plt.title(f"Loss Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    loss_plot_path = os.path.join(output_dir, f"loss_curve_{model_name}.png")
    plt.savefig(loss_plot_path)
    plt.close()


# Run all models
for name, builder in models.items():
    train_and_plot(name, builder)
