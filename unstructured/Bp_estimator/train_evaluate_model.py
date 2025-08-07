import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def create_model(input_shape):
    model = models.Sequential([
        # Use small number of filters and kernel sizes for TinyML
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

def main():
    # Load the prepared dataset
    data = np.load('tiny_dataset.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Load scalers
    with open('scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    bp_scaler = scalers['bp_scaler']
    
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
        verbose=0
    )
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Get predictions in original scale
    y_test_orig = bp_scaler.inverse_transform(y_test)
    y_pred_orig = bp_scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    sbp_diff = y_pred_orig[:, 0] - y_test_orig[:, 0]
    dbp_diff = y_pred_orig[:, 1] - y_test_orig[:, 1]
    
    print("\n=== Clinical Validation Results ===")
    print("\nSystolic Blood Pressure (SBP):")
    print(f"Mean Error: {np.mean(sbp_diff):.2f} ± {np.std(sbp_diff):.2f} mmHg")
    print(f"Within ±5 mmHg: {np.mean(np.abs(sbp_diff) <= 5)*100:.1f}%")
    print(f"Within ±10 mmHg: {np.mean(np.abs(sbp_diff) <= 10)*100:.1f}%")
    print(f"Within ±15 mmHg: {np.mean(np.abs(sbp_diff) <= 15)*100:.1f}%")
    
    print("\nDiastolic Blood Pressure (DBP):")
    print(f"Mean Error: {np.mean(dbp_diff):.2f} ± {np.std(dbp_diff):.2f} mmHg")
    print(f"Within ±5 mmHg: {np.mean(np.abs(dbp_diff) <= 5)*100:.1f}%")
    print(f"Within ±10 mmHg: {np.mean(np.abs(dbp_diff) <= 10)*100:.1f}%")
    print(f"Within ±15 mmHg: {np.mean(np.abs(dbp_diff) <= 15)*100:.1f}%")
    
    # Save the model
    model.save('bp_model.h5')
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open('bp_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"\nTFLite Model Size: {len(tflite_model) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
