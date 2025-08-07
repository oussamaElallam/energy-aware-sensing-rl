# Health Wearable Case Study

This case study demonstrates the energy-aware sensing framework applied to wearable health monitoring using an ESP32-S3 microcontroller.

## Overview

The health wearable monitors three physiological signals:
- **ECG**: Heart rhythm classification (arrhythmia detection)
- **PPG**: Blood pressure estimation 
- **Temperature**: Fever detection

The system uses reinforcement learning to optimize sensor activation patterns, balancing detection accuracy with battery life.

## Hardware Setup

See `hardware_setup.md` for complete wiring instructions.

### Components
- ESP32-S3 Development Board
- AD8232 ECG Sensor
- MAX30105 PPG Sensor  
- MLX90614 Temperature Sensor
- Status LEDs (Green/Red/Yellow)

### Key Specifications
- **Battery Life**: 16+ hours with RL optimization
- **ECG Model**: 63.4KB TFLite model for 5-class arrhythmia detection
- **BP Model**: 73.2KB TFLite model for systolic/diastolic estimation
- **Q-table**: 283KB learned policy for sensor scheduling

## Firmware

The Arduino firmware (`firmware/main.ino`) implements:

1. **Sensor Management**: I2C communication with PPG/temp sensors, ADC sampling for ECG
2. **TinyML Inference**: On-device classification using TensorFlow Lite Micro
3. **RL Policy**: Q-table lookup for sensor activation decisions
4. **Power Management**: Dynamic sensor enable/disable based on battery level and patient state

### Key Features
- Real-time ECG R-peak detection and heart rate calculation
- PPG signal processing with moving average filtering
- Clinical threshold validation for all measurements
- LED status indicators for normal/abnormal/error states

## Training Data

The RL agent was trained on synthetic physiological data with realistic event probabilities:
- Arrhythmia events: 30% probability
- Blood pressure anomalies: 40% probability  
- Fever events: 20% probability

## Results

16-hour evaluation demonstrates:
- **Energy Savings**: 35% reduction in power consumption vs. always-on sensing
- **Detection Accuracy**: >95% for critical events (arrhythmia, severe hypertension)
- **Missed Event Rate**: <2% with risk-aware reward tuning (λ=1.0)

## Usage

1. **Hardware Setup**: Follow `hardware_setup.md` for sensor connections
2. **Firmware Upload**: Use Arduino IDE to upload `firmware/main.ino`
3. **Monitor Output**: Serial console at 115200 baud shows real-time measurements
4. **Status LEDs**: 
   - Green: Normal heart rhythm detected
   - Red: Arrhythmia or abnormal BP detected
   - Yellow: Sensor error or low battery

## Extending the Case Study

To adapt this framework for other health monitoring applications:

1. **New Sensors**: Add sensor drivers and modify I2C scanning in firmware
2. **New Models**: Train TFLite models and convert using `framework/convert_to_c_array.py`
3. **New Events**: Update event flags in training data and reward function
4. **New Policies**: Retrain Q-table using `scripts/train_q_learning.py` with custom parameters
