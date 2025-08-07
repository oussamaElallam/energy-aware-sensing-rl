import wfdb
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_mitbih_data(data_path, n_samples=1000):
    """
    Load MIT-BIH data and extract features
    """
    records = []
    labels = []
    
    # List all .dat files
    record_files = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]
    
    for record_name in record_files[:n_samples]:
        # Read the record
        record = wfdb.rdrecord(os.path.join(data_path, record_name))
        annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')
        
        # Get signal data
        signals = record.p_signal.T[0]  # Using first channel
        
        # Process annotations
        for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
            if symbol in ['N', 'L', 'R', 'A', 'V']:  # Normal, Left bundle branch block, Right bundle branch block, Atrial premature, Premature ventricular contraction
                # Extract a window of data around the annotation
                start_idx = max(0, sample_idx - 90)  # 250ms before at 360Hz
                end_idx = min(len(signals), sample_idx + 90)  # 250ms after
                window = signals[start_idx:end_idx]
                
                if len(window) == 180:  # Only use complete windows
                    records.append(window)
                    labels.append(['N', 'L', 'R', 'A', 'V'].index(symbol))
    
    return np.array(records), np.array(labels)

def preprocess_data(X, y):
    """
    Preprocess the data for model training
    """
    # Normalize the data
    scaler = StandardScaler()
    X_reshaped = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_processed = X_scaled.reshape(X.shape)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    data_path = "mit-bih-arrhythmia-database-1.0.0"
    X, y = load_mitbih_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Save preprocessed data
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
