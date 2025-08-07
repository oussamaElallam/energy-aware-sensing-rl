import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
import os

# --- Model Architecture ---
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(16, 3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    # Create and compile model
    model = create_model((X_train.shape[1], 1), len(np.unique(y_train)))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print("Training ECG classifier model...")
    history = model.fit(
        X_train[..., np.newaxis],
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test[..., np.newaxis], y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the model
    model.save("ecg_model.h5")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    # Save TFLite model
    with open("ecg_model.tflite", "wb") as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize("ecg_model.tflite") / 1024  # Size in KB
    print(f"TFLite model size: {tflite_size:.2f} KB")
