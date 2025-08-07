import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(8, 3, activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(2)  # 2 outputs: SBP and DBP
    ])
    return model

# Load the prepared dataset
data = np.load('tiny_dataset.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Reshape input data for Conv1D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create and compile model
model = create_model((128, 1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save the model
model.save('bp_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('bp_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\nTFLite Model Size: {len(tflite_model) / 1024:.2f} KB")
