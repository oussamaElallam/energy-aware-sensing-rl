import h5py
import numpy as np
import pandas as pd

def analyze_hdf5_dataset(file_path):
    print(f"Analyzing dataset from: {file_path}")
    print("-" * 50)
    
    with h5py.File(file_path, 'r') as f:
        # Print the structure of the HDF5 file
        print("Dataset Structure:")
        print("-" * 20)
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Size (MB): {obj.size * obj.dtype.itemsize / (1024*1024):.2f}")
                
                # Print some basic statistics if the data is numeric
                if np.issubdtype(obj.dtype, np.number):
                    data = obj[:]
                    print(f"  Min: {np.min(data)}")
                    print(f"  Max: {np.max(data)}")
                    print(f"  Mean: {np.mean(data)}")
                    print(f"  Std: {np.std(data)}")
                print()
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        
        f.visititems(print_structure)

if __name__ == "__main__":
    file_path = "small_dataset.hdf5"
    analyze_hdf5_dataset(file_path)
