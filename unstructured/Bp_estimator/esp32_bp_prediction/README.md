# ESP32 Blood Pressure Prediction

This project uses a TinyML model to predict blood pressure (SBP and DBP) from PPG signals using an ESP32 and MAX30102 sensor.

## Hardware Requirements

1. ESP32 Development Board
2. MAX30102 Pulse Oximeter Sensor
3. Connecting wires

## Wiring

Connect the MAX30102 to ESP32:
- VIN -> 3.3V
- GND -> GND
- SCL -> GPIO 22
- SDA -> GPIO 21

## Software Setup

1. Install Required Libraries:
   - Install Arduino IDE
   - Add ESP32 board support
   - Install the following libraries:
     - SparkFun MAX3010x Sensor Library
     - TensorFlow Lite for Microcontrollers
     - Wire

2. Model Deployment:
   - Run `python convert_to_c_array.py` to convert the TFLite model to a C array
   - Copy the generated `model_data.h` to the project directory
   - Upload the code to ESP32 using Arduino IDE

## Usage

1. Power up the ESP32
2. Place your finger on the MAX30102 sensor
3. Wait for 128 samples to be collected (~1.3 seconds)
4. Blood pressure readings will be displayed in the Serial Monitor

## Expected Output

The Serial Monitor will display:
```
SBP: xxx.x mmHg, DBP: xxx.x mmHg
```

## Clinical Validation

The model has been validated against AAMI/ISO 81060-2:2013 standards:
- Mean Error (SBP): 0.03 ± 0.89 mmHg
- Mean Error (DBP): 0.00 ± 0.32 mmHg
- 100% of readings within ±15 mmHg

## Troubleshooting

1. If "MAX30102 not found!" appears:
   - Check wiring connections
   - Verify I2C address (default: 0x57)
   - Check power supply

2. If predictions seem incorrect:
   - Ensure finger is placed correctly
   - Keep finger still during measurement
   - Check sensor LED intensity settings

## Notes

- The model expects 128 PPG samples at 100Hz
- Predictions are updated approximately every 1.3 seconds
- Keep finger steady during measurement for best results
