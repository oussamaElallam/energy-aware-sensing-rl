#include <Wire.h>
#include "MAX30102.h"
#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Constants for PPG signal
#define SIGNAL_SIZE 128
#define SAMPLING_FREQ 100  // Hz
#define SAMPLING_PERIOD_MS (1000 / SAMPLING_FREQ)
#define MOVING_AVG_SIZE 4

// Pin definitions
#define SDA_PIN 21
#define SCL_PIN 22

// Global variables
MAX30102 particleSensor;
float ppg_buffer[SIGNAL_SIZE];
int buffer_index = 0;
uint32_t moving_avg_buffer[MOVING_AVG_SIZE];
int moving_avg_index = 0;
float ppg_min = 50000;  // Dynamic range adjustment
float ppg_max = 0;      // Dynamic range adjustment

// TFLite globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Arena size for TFLite memory arena
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  
  // Initialize MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found!");
    while (1);
  }
  
  // Configure sensor settings
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeIR(0x0A);
  
  // Initialize moving average buffer
  for (int i = 0; i < MOVING_AVG_SIZE; i++) {
    moving_avg_buffer[i] = 0;
  }
  
  // Set up logging
  Serial.println("Initializing TFLite...");
  
  // Map the model into a usable data structure
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
  
  // Create an interpreter to run the model
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory for the model's input and output tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }
  
  // Get pointers for the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("Setup complete");
}

// Calculate moving average
uint32_t calculate_moving_average(uint32_t new_value) {
  uint32_t sum = 0;
  moving_avg_buffer[moving_avg_index] = new_value;
  moving_avg_index = (moving_avg_index + 1) % MOVING_AVG_SIZE;
  
  for (int i = 0; i < MOVING_AVG_SIZE; i++) {
    sum += moving_avg_buffer[i];
  }
  
  return sum / MOVING_AVG_SIZE;
}

// Normalize PPG signal using dynamic range
float normalize_ppg(uint32_t value) {
  // Update min/max values
  if (value < ppg_min) ppg_min = value;
  if (value > ppg_max) ppg_max = value;
  
  // Prevent division by zero
  if (ppg_max == ppg_min) return 0.5f;
  
  // Normalize to 0-1 range
  return (float)(value - ppg_min) / (float)(ppg_max - ppg_min);
}

void loop() {
  static uint32_t last_sample = 0;
  uint32_t current_time = millis();
  
  // Check if it's time to take a new sample
  if (current_time - last_sample >= SAMPLING_PERIOD_MS) {
    last_sample = current_time;
    
    // Check if new data is available
    if (particleSensor.available()) {
      // Read PPG value (using red LED)
      uint32_t raw_red = particleSensor.getRed();
      
      // Apply moving average filter
      uint32_t filtered_red = calculate_moving_average(raw_red);
      
      if (buffer_index < SIGNAL_SIZE) {
        // Store normalized PPG value
        ppg_buffer[buffer_index++] = normalize_ppg(filtered_red);
        
        // Print raw and normalized values for debugging
        Serial.print("Raw: ");
        Serial.print(raw_red);
        Serial.print(", Filtered: ");
        Serial.print(filtered_red);
        Serial.print(", Normalized: ");
        Serial.println(ppg_buffer[buffer_index-1], 4);
        
        // If buffer is full, make prediction
        if (buffer_index == SIGNAL_SIZE) {
          // Copy buffer to input tensor
          for (int i = 0; i < SIGNAL_SIZE; i++) {
            input->data.f[i] = ppg_buffer[i];
          }
          
          // Run inference
          TfLiteStatus invoke_status = interpreter->Invoke();
          if (invoke_status != kTfLiteOk) {
            Serial.println("Invoke failed!");
            return;
          }
          
          // Get predictions and convert to mmHg
          float sbp = output->data.f[0] * 40 + 100;  // Scale to reasonable BP range
          float dbp = output->data.f[1] * 30 + 60;   // Scale to reasonable BP range
          
          // Print results
          Serial.print("SBP: ");
          Serial.print(sbp, 1);
          Serial.print(" mmHg, DBP: ");
          Serial.print(dbp, 1);
          Serial.println(" mmHg");
          
          // Reset buffer index
          buffer_index = 0;
          
          // Reset min/max for next reading
          ppg_min = 50000;
          ppg_max = 0;
        }
      }
    }
  }
}
