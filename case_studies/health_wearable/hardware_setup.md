# Wearable TinyML System - Hardware Setup Guide

This document provides the wiring diagram and hardware setup instructions for the wearable TinyML reinforcement learning system.

## Components List

- ESP32-S3 Development Board
- MAX30105 PPG Sensor (for blood pressure estimation)
- AD8232 ECG Sensor
- MLX90614 Temperature Sensor
- LEDs (for status indication)
  - Green LED (Normal rhythm)
  - Red LED (Abnormal rhythm)
  - Yellow LED (Error status)
- Jumper wires
- Power supply (3.7V LiPo battery recommended)

## Pin Connections

### I2C Configuration
- SDA: GPIO 8
- SCL: GPIO 9

### ECG Sensor (AD8232)
- ECG Output: GPIO 14 (ADC pin)
- LO+ (Lead Off Detection Positive): GPIO 13
- LO- (Lead Off Detection Negative): GPIO 12
- SDN (Shutdown pin): GPIO 15

### LED Indicators
- Normal Rhythm LED (Green): GPIO 4
- Abnormal Rhythm LED (Red): GPIO 5
- Error LED (Yellow): GPIO 2

### Sensor I2C Addresses
- MAX30105 (PPG Sensor): 0x57
- MLX90614 (Temperature Sensor): 0x5A

## Wiring Diagram

```
ESP32-S3                      MAX30105 (PPG)
---------                     -------------
GPIO 8 (SDA) --------------> SDA
GPIO 9 (SCL) --------------> SCL
3.3V        --------------> VIN
GND         --------------> GND

ESP32-S3                      MLX90614 (Temp)
---------                     ---------------
GPIO 8 (SDA) --------------> SDA
GPIO 9 (SCL) --------------> SCL
3.3V        --------------> VIN
GND         --------------> GND

ESP32-S3                      AD8232 (ECG)
---------                     ------------
GPIO 14      --------------> OUTPUT
GPIO 13      --------------> LO+
GPIO 12      --------------> LO-
GPIO 15      --------------> SDN
3.3V         --------------> 3.3V
GND          --------------> GND

ESP32-S3                      Status LEDs
---------                     ----------
GPIO 4       --------------> Green LED (+ resistor) --> GND
GPIO 5       --------------> Red LED (+ resistor) --> GND
GPIO 2       --------------> Yellow LED (+ resistor) --> GND
```

## Power Considerations

The system is designed to operate on battery power and uses reinforcement learning to optimize power consumption. The firmware implements power management strategies through selectively enabling/disabling sensors based on the patient's condition and battery level.

- Battery voltage should be 3.3-5V
- Connect battery positive to VIN and negative to GND
- Estimated runtime varies based on sensor usage pattern

## Testing the Setup

1. After connecting the components, upload the firmware to the ESP32-S3
2. Open the Serial Monitor at 115200 baud
3. The system will scan the I2C bus and report found devices
4. Verify that all sensors are properly detected
5. The system will begin monitoring according to the reinforcement learning policy

## Troubleshooting

- If the yellow LED blinks continuously, check I2C connections
- If you see "MAX30102 not found" in serial output, check PPG sensor wiring
- If ECG readings show flat line, ensure proper electrode placement
