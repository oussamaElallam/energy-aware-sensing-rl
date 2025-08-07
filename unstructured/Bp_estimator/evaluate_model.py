import numpy as np
import tensorflow as tf
import pickle
import os
import sys

# Redirect TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all logging
tf.get_logger().setLevel('ERROR')  # Only show errors

def evaluate_bp_model():
    try:
        # Load test data
        data = np.load('tiny_dataset.npz')
        X_test = data['X_test'].reshape(-1, 128, 1)
        y_test = data['y_test']

        # Load model and scalers
        model = tf.keras.models.load_model('bp_model.h5', compile=False)
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        bp_scaler = scalers['bp_scaler']

        # Make predictions
        y_pred = model.predict(X_test, verbose=0)

        # Convert to original scale
        y_test_orig = bp_scaler.inverse_transform(y_test)
        y_pred_orig = bp_scaler.inverse_transform(y_pred)

        # Calculate errors
        sbp_errors = y_pred_orig[:, 0] - y_test_orig[:, 0]
        dbp_errors = y_pred_orig[:, 1] - y_test_orig[:, 1]

        # Print results directly
        print("\n=== Clinical Validation Results ===")
        print("\nAAMI/ISO 81060-2:2013 Standard Requirements:")
        print("- Mean difference ≤ 5 mmHg")
        print("- Standard deviation ≤ 8 mmHg")
        print("- 85% of readings within ±15 mmHg")

        print("\nSystolic Blood Pressure (SBP):")
        print(f"Mean Error: {np.mean(sbp_errors):.2f} ± {np.std(sbp_errors):.2f} mmHg")
        print(f"Within ±5 mmHg: {np.mean(np.abs(sbp_errors) <= 5)*100:.1f}%")
        print(f"Within ±10 mmHg: {np.mean(np.abs(sbp_errors) <= 10)*100:.1f}%")
        print(f"Within ±15 mmHg: {np.mean(np.abs(sbp_errors) <= 15)*100:.1f}%")

        print("\nDiastolic Blood Pressure (DBP):")
        print(f"Mean Error: {np.mean(dbp_errors):.2f} ± {np.std(dbp_errors):.2f} mmHg")
        print(f"Within ±5 mmHg: {np.mean(np.abs(dbp_errors) <= 5)*100:.1f}%")
        print(f"Within ±10 mmHg: {np.mean(np.abs(dbp_errors) <= 10)*100:.1f}%")
        print(f"Within ±15 mmHg: {np.mean(np.abs(dbp_errors) <= 15)*100:.1f}%")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    evaluate_bp_model()
