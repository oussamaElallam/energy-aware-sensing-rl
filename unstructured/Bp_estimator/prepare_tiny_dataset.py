import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

def extract_bp_features(abp_signal):
    # Extract Systolic (max) and Diastolic (min) blood pressure
    sbp = np.max(abp_signal)
    dbp = np.min(abp_signal)
    return sbp, dbp

def prepare_tiny_dataset(input_file, output_file, n_samples=1000, n_points=128):
    with h5py.File(input_file, 'r') as f:
        # Get original PPG and ABP data
        ppg_data = f['resampled/ppg'][:]
        abp_data = f['resampled/abp'][:]
        
        # Randomly select n_samples
        indices = np.random.choice(ppg_data.shape[0], n_samples, replace=False)
        ppg_selected = ppg_data[indices]
        abp_selected = abp_data[indices]
        
        # Reduce the number of points per signal using interpolation
        x_original = np.linspace(0, 1, ppg_selected.shape[1])
        x_new = np.linspace(0, 1, n_points)
        
        ppg_reduced = np.array([np.interp(x_new, x_original, sig) for sig in ppg_selected])
        
        # Extract SBP and DBP from ABP signals
        sbp_dbp = np.array([extract_bp_features(sig) for sig in abp_selected])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            ppg_reduced, sbp_dbp, test_size=0.2, random_state=42
        )
        
        # Normalize the data
        ppg_scaler = MinMaxScaler()
        bp_scaler = MinMaxScaler()
        
        X_train_scaled = ppg_scaler.fit_transform(X_train)
        X_test_scaled = ppg_scaler.transform(X_test)
        
        y_train_scaled = bp_scaler.fit_transform(y_train)
        y_test_scaled = bp_scaler.transform(y_test)
        
        # Save the processed data
        np.savez(output_file,
                 X_train=X_train_scaled,
                 X_test=X_test_scaled,
                 y_train=y_train_scaled,
                 y_test=y_test_scaled)
        
        # Save the scalers for later use
        with open('scalers.pkl', 'wb') as f:
            pickle.dump({
                'ppg_scaler': ppg_scaler,
                'bp_scaler': bp_scaler
            }, f)
        
        print(f"Dataset reduced from {ppg_data.shape} to {X_train_scaled.shape} (training set)")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        # Calculate and print approximate file sizes
        original_size = ppg_data.nbytes / (1024 * 1024)
        reduced_size = (X_train_scaled.nbytes + X_test_scaled.nbytes + 
                       y_train_scaled.nbytes + y_test_scaled.nbytes) / (1024 * 1024)
        print(f"\nOriginal dataset size: {original_size:.2f} MB")
        print(f"Reduced dataset size: {reduced_size:.2f} MB")

if __name__ == "__main__":
    prepare_tiny_dataset(
        input_file="small_dataset.hdf5",
        output_file="tiny_dataset.npz",
        n_samples=1000,  # Using 1000 samples
        n_points=128     # Reducing to 128 points per signal
    )
