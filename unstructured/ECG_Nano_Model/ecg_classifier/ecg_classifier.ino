#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// AD8232 pin configuration
const int LO_PLUS = 13;    // LO+ pin
const int LO_MINUS = 12;   // LO- pin
const int OUTPUT_PIN = 14; // Output pin

// LED indicators
const int ERROR_LED = 2;     // Red LED for errors
const int NORMAL_LED = 4;    // Green LED for normal beats
const int ABNORMAL_LED = 5;  // Yellow LED for abnormal beats

// Constants for ECG processing
const int SAMPLING_FREQ = 360;  // 360Hz to match MIT-BIH
const int WINDOW_SIZE = 180;    // 500ms window
const int INPUT_BUFFER_SIZE = WINDOW_SIZE;

// Buffer for ECG samples
float ecg_buffer[INPUT_BUFFER_SIZE];
int buffer_index = 0;

// Global variables for TensorFlow Lite
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  
  // Arena size for TensorFlow Lite memory arena
  constexpr int kTensorArenaSize = 32 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// Clinical thresholds
const float MIN_QRS_DURATION = 80.0;  // ms
const float MAX_QRS_DURATION = 120.0; // ms
const float MIN_ST_ELEVATION = -0.1;  // mV
const float MAX_ST_ELEVATION = 0.2;   // mV
const float MIN_HEART_RATE = 40.0;    // BPM
const float MAX_HEART_RATE = 200.0;   // BPM

// Structure to hold ECG metrics with validation
struct ECGMetrics {
    float heart_rate;
    const char* rhythm_class;
    float qrs_duration;
    float st_elevation;
    bool valid_hr;
    bool valid_qrs;
    bool valid_st;
    float confidence_score;
};

// Buffer for R-peak detection
const int MAX_PEAKS = 10;
float r_peak_timestamps[MAX_PEAKS];
int peak_count = 0;
unsigned long last_peak_time = 0;
float last_r_peak_value = 0;

// Thresholds for R-peak detection
const float PEAK_THRESHOLD = 0.7;  // Normalized threshold
const int MIN_RR_SAMPLES = 108;    // Minimum 300ms at 360Hz

// Class labels
const char* LABELS[] = {"Normal", "Left BBB", "Right BBB", "Atrial Premature", "PVC"};

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Configure AD8232 pins
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  pinMode(OUTPUT_PIN, INPUT);
  
  // Configure LED pins
  pinMode(ERROR_LED, OUTPUT);
  pinMode(NORMAL_LED, OUTPUT);
  pinMode(ABNORMAL_LED, OUTPUT);
  
  // Initial LED test
  digitalWrite(ERROR_LED, HIGH);
  digitalWrite(NORMAL_LED, HIGH);
  digitalWrite(ABNORMAL_LED, HIGH);
  delay(1000);
  digitalWrite(ERROR_LED, LOW);
  digitalWrite(NORMAL_LED, LOW);
  digitalWrite(ABNORMAL_LED, LOW);
  
  // Set up logging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  // Map the model into a usable data structure
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version mismatch!");
    return;
  }
  
  // This pulls in the operators implementations we need
  static tflite::AllOpsResolver resolver;
  
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }
  
  // Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // Initialize the ECG buffer
  for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
    ecg_buffer[i] = 0.0;
  }
  
  Serial.println("ECG Classifier initialized!");
}

// Normalize the ECG data
void normalize_data() {
  float mean = 0.0;
  float std = 0.0;
  
  // Calculate mean
  for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
    mean += ecg_buffer[i];
  }
  mean /= INPUT_BUFFER_SIZE;
  
  // Calculate standard deviation
  for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
    std += (ecg_buffer[i] - mean) * (ecg_buffer[i] - mean);
  }
  std = sqrt(std / INPUT_BUFFER_SIZE);
  
  // Normalize the data
  for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
    ecg_buffer[i] = (ecg_buffer[i] - mean) / (std + 1e-6);
  }
}

// Perform inference on the ECG window
void classify_ecg() {
  // Turn off all LEDs before classification
  digitalWrite(NORMAL_LED, LOW);
  digitalWrite(ABNORMAL_LED, LOW);
  
  // Normalize the data
  normalize_data();
  
  // Copy the ECG data to the input tensor
  for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
    input->data.f[i] = ecg_buffer[i];
  }
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed!");
    digitalWrite(ERROR_LED, HIGH);
    return;
  }
  
  // Get the output
  float* results = output->data.f;
  
  // Find the class with highest probability
  float max_prob = 0.0;
  int max_index = 0;
  for (int i = 0; i < 5; i++) {
    if (results[i] > max_prob) {
      max_prob = results[i];
      max_index = i;
    }
  }
  
  // Only classify if confidence is above threshold
  if (max_prob < 0.7) {
    Serial.println("Low confidence, skipping classification");
    return;
  }
  
  // Update LEDs based on classification
  if (max_index == 0) { // Normal beat
    digitalWrite(NORMAL_LED, HIGH);
    digitalWrite(ABNORMAL_LED, LOW);
  } else { // Abnormal beat
    digitalWrite(NORMAL_LED, LOW);
    digitalWrite(ABNORMAL_LED, HIGH);
  }
  
  // Print the result
  Serial.print("Classification: ");
  Serial.print(LABELS[max_index]);
  Serial.print(" (Confidence: ");
  Serial.print(max_prob * 100);
  Serial.println("%)");
}

// Detect R-peaks in the signal
bool detectRPeak(float current_sample, unsigned long current_time) {
    static float max_value = 0;
    static int samples_since_last_peak = 0;
    
    samples_since_last_peak++;
    
    // Update maximum value
    if (current_sample > max_value) {
        max_value = current_sample;
    }
    
    // Check if this is a peak
    if (current_sample > PEAK_THRESHOLD && 
        current_sample > last_r_peak_value && 
        samples_since_last_peak >= MIN_RR_SAMPLES) {
        
        // Store peak information
        if (peak_count < MAX_PEAKS) {
            r_peak_timestamps[peak_count++] = current_time;
        } else {
            // Shift array and add new peak
            for (int i = 0; i < MAX_PEAKS - 1; i++) {
                r_peak_timestamps[i] = r_peak_timestamps[i + 1];
            }
            r_peak_timestamps[MAX_PEAKS - 1] = current_time;
        }
        
        last_peak_time = current_time;
        last_r_peak_value = current_sample;
        samples_since_last_peak = 0;
        max_value = 0;
        return true;
    }
    
    return false;
}

// Calculate heart rate from R-peak intervals
float calculateHeartRate() {
    if (peak_count < 2) return 0;
    
    float total_interval = 0;
    int intervals = 0;
    
    // Calculate average interval between peaks
    for (int i = 1; i < peak_count; i++) {
        float interval = (r_peak_timestamps[i] - r_peak_timestamps[i-1]) / 1000.0; // Convert to seconds
        total_interval += interval;
        intervals++;
    }
    
    if (intervals == 0) return 0;
    
    float avg_interval = total_interval / intervals;
    return 60.0 / avg_interval; // Convert to BPM
}

// Improved QRS duration measurement using Pan-Tompkins approach
float measureQRSDuration(float* signal, int peak_index) {
    // Parameters for QRS detection
    const float HIGH_PASS_ALPHA = 0.95;  // High-pass filter coefficient
    const float LOW_PASS_ALPHA = 0.15;   // Low-pass filter coefficient
    
    // Buffers for filtered signals
    float filtered[INPUT_BUFFER_SIZE];
    float derivative[INPUT_BUFFER_SIZE];
    float squared[INPUT_BUFFER_SIZE];
    
    // High-pass filter
    for (int i = 1; i < INPUT_BUFFER_SIZE; i++) {
        filtered[i] = HIGH_PASS_ALPHA * (filtered[i-1] + signal[i] - signal[i-1]);
    }
    
    // Low-pass filter
    for (int i = 1; i < INPUT_BUFFER_SIZE; i++) {
        filtered[i] = filtered[i] + LOW_PASS_ALPHA * (filtered[i-1] - filtered[i]);
    }
    
    // Derivative
    for (int i = 2; i < INPUT_BUFFER_SIZE-2; i++) {
        derivative[i] = (2*filtered[i+2] + filtered[i+1] - filtered[i-1] - 2*filtered[i-2]) / 8.0;
    }
    
    // Square
    for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
        squared[i] = derivative[i] * derivative[i];
    }
    
    // Find QRS onset and offset using adaptive threshold
    float max_energy = 0;
    for (int i = peak_index-20; i <= peak_index+20 && i < INPUT_BUFFER_SIZE; i++) {
        if (i >= 0 && squared[i] > max_energy) {
            max_energy = squared[i];
        }
    }
    
    float threshold = max_energy * 0.15;  // 15% of max energy
    
    // Find QRS onset
    int onset = peak_index;
    for (int i = peak_index; i >= 0 && i > peak_index - 40; i--) {
        if (squared[i] < threshold) {
            onset = i;
            break;
        }
    }
    
    // Find QRS offset
    int offset = peak_index;
    for (int i = peak_index; i < INPUT_BUFFER_SIZE && i < peak_index + 40; i++) {
        if (squared[i] < threshold) {
            offset = i;
            break;
        }
    }
    
    // Convert samples to milliseconds
    return (offset - onset) * (1000.0 / SAMPLING_FREQ);
}

// Improved ST segment measurement with baseline correction
float measureSTElevation(float* signal, int peak_index) {
    // Parameters
    const int PR_SEGMENT_START = 50;  // samples before R peak
    const int ST_SEGMENT_START = 80;  // samples after R peak
    const int SEGMENT_LENGTH = 10;    // samples to average
    
    float pr_baseline = 0;
    float st_level = 0;
    int pr_samples = 0;
    int st_samples = 0;
    
    // Calculate PR baseline (reference level)
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        int idx = peak_index - PR_SEGMENT_START + i;
        if (idx >= 0 && idx < INPUT_BUFFER_SIZE) {
            pr_baseline += signal[idx];
            pr_samples++;
        }
    }
    
    // Calculate ST level
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        int idx = peak_index + ST_SEGMENT_START + i;
        if (idx >= 0 && idx < INPUT_BUFFER_SIZE) {
            st_level += signal[idx];
            st_samples++;
        }
    }
    
    // Avoid division by zero
    if (pr_samples == 0 || st_samples == 0) return 0;
    
    pr_baseline /= pr_samples;
    st_level /= st_samples;
    
    // Return ST deviation from baseline
    return st_level - pr_baseline;
}

// Format metrics as JSON
void formatMetricsJSON(ECGMetrics metrics) {
    Serial.print("{\"ecg_metrics\":{");
    Serial.print("\"heart_rate\":");
    Serial.print(metrics.heart_rate);
    Serial.print(",\"rhythm_class\":\"");
    Serial.print(metrics.rhythm_class);
    Serial.print("\",\"qrs_duration\":");
    Serial.print(metrics.qrs_duration);
    Serial.print(",\"st_elevation\":");
    Serial.print(metrics.st_elevation);
    Serial.print(",\"valid_hr\":");
    Serial.print(metrics.valid_hr);
    Serial.print(",\"valid_qrs\":");
    Serial.print(metrics.valid_qrs);
    Serial.print(",\"valid_st\":");
    Serial.print(metrics.valid_st);
    Serial.print(",\"confidence_score\":");
    Serial.print(metrics.confidence_score);
    Serial.println("}}");
}

void loop() {
  // Check if leads are properly connected
  if ((digitalRead(LO_PLUS) == 1) || (digitalRead(LO_MINUS) == 1)) {
    Serial.println("Leads off!");
    digitalWrite(ERROR_LED, HIGH);
    delay(500);
    return;
  }
  digitalWrite(ERROR_LED, LOW);
  
  // Read ECG value with error checking
  int raw_value = analogRead(OUTPUT_PIN);
  if (raw_value < 0 || raw_value > 4095) { // ESP32 ADC range
    Serial.println("Invalid ADC reading!");
    digitalWrite(ERROR_LED, HIGH);
    return;
  }
  
  // Convert to float and normalize to 0-1 range
  float ecg_value = raw_value / 4095.0;
  
  // Add to buffer
  ecg_buffer[buffer_index] = ecg_value;
  
  // Check for R-peak
  if (detectRPeak(ecg_value, millis())) {
    // R-peak detected, can calculate features here if needed
  }
  
  buffer_index++;
  
  // When buffer is full, perform analysis
  if (buffer_index >= INPUT_BUFFER_SIZE) {
    // Create metrics structure
    ECGMetrics metrics;
    
    // 1. Calculate heart rate
    metrics.heart_rate = calculateHeartRate();
    metrics.valid_hr = (metrics.heart_rate >= MIN_HEART_RATE && metrics.heart_rate <= MAX_HEART_RATE);
    
    // 2. Perform classification
    normalize_data();
    
    // Copy the ECG data to the input tensor
    for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
      input->data.f[i] = ecg_buffer[i];
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed!");
      digitalWrite(ERROR_LED, HIGH);
      return;
    }
    
    // Get the output
    float* results = output->data.f;
    
    // Find the class with highest probability
    float max_prob = 0.0;
    int max_index = 0;
    for (int i = 0; i < 5; i++) {
      if (results[i] > max_prob) {
        max_prob = results[i];
        max_index = i;
      }
    }
    
    // Only proceed if confidence is above threshold
    if (max_prob >= 0.7) {
      metrics.rhythm_class = LABELS[max_index];
      metrics.confidence_score = max_prob;
      
      // 3. Calculate QRS duration
      // Find the highest peak in the current window
      int peak_index = 0;
      float max_value = 0;
      for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
        if (ecg_buffer[i] > max_value) {
          max_value = ecg_buffer[i];
          peak_index = i;
        }
      }
      metrics.qrs_duration = measureQRSDuration(ecg_buffer, peak_index);
      metrics.valid_qrs = (metrics.qrs_duration >= MIN_QRS_DURATION && metrics.qrs_duration <= MAX_QRS_DURATION);
      
      // 4. Measure ST elevation
      metrics.st_elevation = measureSTElevation(ecg_buffer, peak_index);
      metrics.valid_st = (metrics.st_elevation >= MIN_ST_ELEVATION && metrics.st_elevation <= MAX_ST_ELEVATION);
      
      // Update LEDs based on classification
      if (max_index == 0) { // Normal beat
        digitalWrite(NORMAL_LED, HIGH);
        digitalWrite(ABNORMAL_LED, LOW);
      } else { // Abnormal beat
        digitalWrite(NORMAL_LED, LOW);
        digitalWrite(ABNORMAL_LED, HIGH);
      }
      
      // Output JSON formatted data
      formatMetricsJSON(metrics);
    }
    
    // Shift the buffer by half window size to maintain overlap
    for (int i = 0; i < INPUT_BUFFER_SIZE/2; i++) {
      ecg_buffer[i] = ecg_buffer[i + INPUT_BUFFER_SIZE/2];
    }
    buffer_index = INPUT_BUFFER_SIZE/2;
  }
  
  // Maintain sampling rate of 360Hz
  delay(1000/SAMPLING_FREQ);
}
