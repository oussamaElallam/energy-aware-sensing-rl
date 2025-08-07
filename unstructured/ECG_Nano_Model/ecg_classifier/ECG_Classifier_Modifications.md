# ECG Classifier Modifications Report
> This document outlines the necessary modifications to align the ECG classifier with Level 2 integration requirements.

## 1. Required Feature Extraction Additions

### A. Current Features
- Rhythm classification (5 classes)
- Basic signal normalization

### B. Features to Add
1. **Heart Rate Calculation**
```cpp
struct ECGMetrics {
    float heart_rate;        // BPM
    String rhythm_class;     // Current implementation
    float qrs_duration;      // milliseconds
    float st_elevation;      // millivolts
};
```

2. **QRS Analysis Function**
```cpp
// Need to implement:
- R-peak detection
- QRS complex duration measurement
- QRS onset and offset detection
```

3. **ST Segment Analysis**
```cpp
// Need to implement:
- ST segment identification
- ST elevation measurement
- J-point detection
```

## 2. Data Integration Requirements

### A. Current Output
```cpp
// Current simple serial output
Serial.print("Classification: ");
Serial.print(LABELS[max_index]);
Serial.print(" (Confidence: ");
Serial.print(max_prob * 100);
Serial.println("%)");
```

### B. Required JSON Output Format
```cpp
// Need to implement this format:
{
    "ecg_metrics": {
        "heart_rate": 75.5,
        "rhythm_class": "Normal",
        "qrs_duration": 95.0,
        "st_elevation": 0.1
    }
}
```

## 3. Implementation Plan

### Phase 1: Core Feature Extraction
1. Add R-peak detection algorithm
2. Implement heart rate calculation
3. Add QRS complex analysis
4. Implement ST segment analysis

### Phase 2: Data Integration
1. Create JSON formatting function
2. Implement data buffering for continuous monitoring
3. Add error checking and validation
4. Implement data transmission protocol

### Phase 3: Optimization
1. Memory usage optimization
2. Processing speed optimization
3. Power consumption optimization

## 4. Code Structure Changes

### A. New Required Functions
```cpp
// 1. R-peak Detection
float detectRPeaks(float* signal, int length);

// 2. Heart Rate Calculation
float calculateHeartRate(float* r_peak_intervals, int length);

// 3. QRS Analysis
float measureQRSDuration(float* signal, int r_peak_index);

// 4. ST Segment Analysis
float measureSTElevation(float* signal, int r_peak_index);

// 5. JSON Formatting
void formatMetricsJSON(ECGMetrics metrics);
```

### B. Modified Main Loop Structure
```cpp
void loop() {
    // Current lead-off detection
    // Current signal acquisition
    
    // New additions needed:
    1. R-peak detection
    2. Heart rate calculation
    3. QRS analysis
    4. ST segment analysis
    5. JSON formatting
    6. Data transmission
}
```

## 5. Memory Considerations

### Current Usage
- TensorFlow Arena: 32KB
- Input Buffer: 180 samples × 4 bytes = 720 bytes

### Additional Requirements
- Peak detection buffer: ~500 bytes
- Feature calculation buffers: ~1KB
- JSON formatting buffer: ~256 bytes

Total Additional Memory: ~2KB

## 6. Performance Requirements

1. **Timing Constraints**
- Maintain 360Hz sampling rate
- Complete all calculations within 2.77ms (1/360 sec)
- JSON output at least every 500ms

2. **Accuracy Requirements**
- Heart rate accuracy: ±5 BPM
- QRS duration accuracy: ±10ms
- ST elevation accuracy: ±0.1mV

## 7. Next Steps

1. **Immediate Actions**
   - Implement R-peak detection
   - Add heart rate calculation
   - Create basic JSON output structure

2. **Short-term Goals**
   - Implement QRS analysis
   - Add ST segment analysis
   - Test and validate all measurements

3. **Long-term Goals**
   - Optimize memory usage
   - Improve processing efficiency
   - Enhance error handling

## Additional Notes
- All modifications should maintain compatibility with the existing TensorFlow Lite model
- Keep the current error handling and LED indication system
- Ensure all new features work within the ESP32's processing capabilities
- Consider implementing a task scheduler if timing becomes critical
