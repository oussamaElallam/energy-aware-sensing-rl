import os
def dump_to_c_array(input_tflite_file, output_header_file):
    print(f"Converting {input_tflite_file} to C array...")
    
    with open(input_tflite_file, 'rb') as f:
        model_data = f.read()
    
    print(f"Model size: {len(model_data)} bytes")
    print(f"First 16 bytes: {model_data[:16].hex()}")
    
    # Create the output C file
    with open('model_data.cpp', 'w') as f:
        f.write('#include "model_data.h"\n\n')
        f.write('// This is a TensorFlow Lite model file that has been converted to a C array.\n\n')
        f.write('alignas(16) ')  # Ensure 16-byte alignment
        f.write('const unsigned char model_tflite[] = {')
        
        # Write the array values 16 bytes per line
        for i in range(0, len(model_data), 16):
            if i % 16 == 0:
                f.write('\n    ')
            chunk = model_data[i:i + 16]
            for byte in chunk:
                f.write(f'0x{byte:02x}, ')
        
        # Close the array
        f.write('\n};\n\n')
        f.write(f'const unsigned int model_tflite_len = {len(model_data)};\n')
    
    print("Conversion completed successfully!")
    print(f"Generated model_data.cpp with size: {len(model_data)} bytes")

# Run the conversion
dump_to_c_array('ecg_model_quantized.tflite', 'model_data.cpp')
'''
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='ecg_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:")
print(f"Shape: {input_details[0]['shape']}")
print(f"Type: {input_details[0]['dtype']}")
print(f"Quantization: {input_details[0]['quantization']}")

print("\nOutput details:")
print(f"Shape: {output_details[0]['shape']}")
print(f"Type: {output_details[0]['dtype']}")
print(f"Quantization: {output_details[0]['quantization']}") '''
'''
import tensorflow as tf
import numpy as np

def representative_dataset():
    # Create representative data matching your input shape [1, 180, 1]
    for _ in range(100):
        data = np.random.uniform(-1, 1, (1, 180, 1)).astype(np.float32)
        yield [data]

def quantize_model(h5_path, tflite_output_path):
    print("Loading model from:", h5_path)
    model = tf.keras.models.load_model(h5_path)
    
    print("Converting to TFLite with quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configure quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    
    try:
        quantized_tflite_model = converter.convert()
        print("Conversion successful!")
        
        # Save the model
        with open(tflite_output_path, 'wb') as f:
            f.write(quantized_tflite_model)
        print(f"Saved quantized model to: {tflite_output_path}")
        
        # Verify the model
        print("\nVerifying model...")
        interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Input details:", input_details)
        print("Output details:", output_details)
        
        return True
        
    except Exception as e:
        print("Error during conversion:", str(e))
        return False

if __name__ == '__main__':
    h5_path = 'ecg_model.h5'  # Replace with your .h5 model path
    tflite_output_path = 'ecg_model_quantized.tflite'
    
    success = quantize_model(h5_path, tflite_output_path)
    if success:
        print("Quantization process completed successfully!")
    else:
        print("Quantization process failed!")'''
        
'''
import tensorflow as tf
import numpy as np

def check_model(tflite_path):
    print(f"\nChecking model: {tflite_path}")
    
    # Read the file
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    print(f"File size: {len(data)} bytes")
    print(f"First 16 bytes: {data[:16].hex()}")
    
    # Try to load with TFLite
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\nModel loaded successfully!")
        print("Input details:", input_details)
        print("Output details:", output_details)
        return True
        
    except Exception as e:
        print("\nError loading model:", str(e))
        return False

def requantize_model(h5_path, tflite_output_path):
    print(f"\nRequantizing model from {h5_path}")
    
    # Load the original model
    model = tf.keras.models.load_model(h5_path)
    
    # Representative dataset generator
    def representative_dataset():
        for _ in range(100):
            data = np.random.uniform(-1, 1, (1, 180, 1)).astype(np.float32)
            yield [data]
    
    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    
    try:
        quantized_tflite_model = converter.convert()
        print("Conversion successful!")
        
        # Save the model
        with open(tflite_output_path, 'wb') as f:
            f.write(quantized_tflite_model)
        print(f"Saved to: {tflite_output_path}")
        
        return check_model(tflite_output_path)
        
    except Exception as e:
        print("Error during conversion:", str(e))
        return False

if __name__ == '__main__':
    # First check the existing model
    existing_model = 'ecg_model_quantized.tflite'  # Your current TFLite model
    if check_model(existing_model):
        print("\nExisting model is valid!")
    else:
        print("\nExisting model needs to be requantized")
        
        # Ask for the original .h5 model path
        h5_path = input("\nPlease enter the path to your original .h5 model: ")
        if not h5_path:
            print("No path provided. Exiting.")
            exit(1)
        
        # Requantize the model
        success = requantize_model(h5_path, 'ecg_model_fixed.tflite')
        if success:
            print("\nModel has been successfully requantized!")
        else:
            print("\nRequantization failed!")'''