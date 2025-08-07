# ECG Classifier for ESP32

## Overview
This is a TinyML implementation of an ECG classifier running on ESP32. It's part of the Level 1 (Nano-TinyML) layer of our hierarchical health monitoring system.

## Features
- Real-time ECG signal processing
- Heart rhythm classification (5 classes)
- QRS complex analysis
- ST segment analysis
- Heart rate calculation
- JSON-formatted output for Level 2 integration

## Hardware Requirements
- ESP32 Development Board
- AD8232 ECG Sensor
- LEDs for status indication:
  * Red LED (Pin 2): Error indicator
  * Green LED (Pin 4): Normal rhythm
  * Yellow LED (Pin 5): Abnormal rhythm

## Pin Configuration
```
AD8232 Connections:
- LO+ → GPIO 13
- LO- → GPIO 12
- OUTPUT → GPIO 14

LED Connections:
- Error LED → GPIO 2
- Normal LED → GPIO 4
- Abnormal LED → GPIO 5
```

## Dependencies
Required Arduino libraries (see requirements.txt):
- TensorFlowLite_ESP32
- Arduino JSON

## Installation
1. Install required libraries using Arduino Library Manager
2. Upload the code to ESP32
3. Connect AD8232 and LEDs according to pin configuration
4. Open Serial Monitor at 115200 baud to view results

## Output Format
The classifier outputs JSON-formatted data:
```json
{
    "ecg_metrics": {
        "heart_rate": float,
        "rhythm_class": string,
        "qrs_duration": float,
        "st_elevation": float,
        "valid_hr": boolean,
        "valid_qrs": boolean,
        "valid_st": boolean,
        "confidence_score": float
    }
}
```

## Clinical Thresholds
- Heart Rate: 40-200 BPM
- QRS Duration: 80-120 ms
- ST Elevation: -0.1 to 0.2 mV

## Model Details
- Input: 180 samples (500ms window at 360Hz)
- Output: 5 classes (Normal, LBBB, RBBB, PVC, APB)
- Model Size: ~64KB
- Memory Usage: 32KB tensor arena

## Files Description
- `ecg_classifier.ino`: Main implementation
- `model_data.cpp/h`: TFLite model data
- `ecg_model.tflite`: TensorFlow Lite model
- `convert_to_c_array.py`: Model conversion utility

## Integration with Level 2
This classifier provides all required metrics for the Level 2 master integration model, following the project's hierarchical architecture.
