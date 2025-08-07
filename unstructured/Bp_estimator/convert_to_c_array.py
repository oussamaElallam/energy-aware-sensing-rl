def convert_to_c_array(input_file, output_file):
    with open(input_file, 'rb') as f:
        model_data = f.read()
    
    c_array = 'const unsigned char model_data[] = {'
    c_array += ','.join([f'0x{b:02x}' for b in model_data])
    c_array += '};\n'
    c_array += f'const unsigned int model_data_len = {len(model_data)};\n'
    
    with open(output_file, 'w') as f:
        f.write(c_array)

if __name__ == '__main__':
    convert_to_c_array('bp_model.tflite', 'model_data.h')
