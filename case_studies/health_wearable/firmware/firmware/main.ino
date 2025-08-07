#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include <Wire.h>
#include "MAX30105.h"
#include "model_bp_data.h"  // BP estimation model data
#include "model_ecg_data.h"  // ECG model data
#include <Adafruit_MLX90614.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "qtable_sensors_time.h" // Q-table for the RL agent

// Constants for PPG signal
#define PPG_SIGNAL_SIZE 128
#define PPG_SAMPLING_FREQ 100  // Hz
#define PPG_SAMPLING_PERIOD_MS (1000 / PPG_SAMPLING_FREQ)
#define MOVING_AVG_SIZE 4

// Constants for ECG
#define ECG_SIGNAL_SIZE 180    // 500ms window at 360Hz
#define ECG_SAMPLING_FREQ 360  // Hz to match MIT-BIH database
#define ECG_SAMPLING_PERIOD_MS (1000 / ECG_SAMPLING_FREQ)
#define ECG_OUTPUT_PIN 14      // ADC pin for ECG
#define LO_PLUS 13            // ECG Lead-off detection positive
#define LO_MINUS 12           // ECG Lead-off detection negative
#define NORMAL_LED 4          // Green LED for normal rhythm
#define ABNORMAL_LED 5        // Red LED for abnormal rhythm
#define ERROR_LED 2           // Yellow LED for errors

// Pin definitions for ECG control
#define ECG_SDN_PIN 15        // Shutdown pin for AD8232

// Pin definitions for I2C
#define I2C_SDA 8         // I2C Data pin
#define I2C_SCL 9         // I2C Clock pin

// I2C Addresses (default addresses for both sensors)
#define MAX30105_I2C_ADDR 0x57
#define MLX90614_I2C_ADDR 0x5A

// Global variables for PPG
MAX30105 particleSensor;
float ppg_buffer[PPG_SIGNAL_SIZE];
uint32_t moving_avg_buffer[MOVING_AVG_SIZE];
int ppg_buffer_index = 0;
int moving_avg_index = 0;
float ppg_min = 50000;  // Dynamic range adjustment
float ppg_max = 0;      // Dynamic range adjustment

// Global variables for ECG
float ecg_buffer[ECG_SIGNAL_SIZE];
int ecg_buffer_index = 0;
const char* ECG_LABELS[] = {"Normal", "Left BBB", "Right BBB", "Atrial Premature", "PVC"};

// MLX90614 temperature sensor
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

// TFLite globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;

// BP estimation model
extern const unsigned char model_data[];  // From BP estimation/model_data.h
const tflite::Model* ppg_model = nullptr;
tflite::MicroInterpreter* ppg_interpreter = nullptr;
TfLiteTensor* ppg_input = nullptr;
TfLiteTensor* ppg_output = nullptr;

// ECG classification model
extern const unsigned char model_tflite[];  // From ECG/model_data.h
extern const unsigned int model_tflite_len;
const tflite::Model* ecg_model = nullptr;
tflite::MicroInterpreter* ecg_interpreter = nullptr;
TfLiteTensor* ecg_input = nullptr;
TfLiteTensor* ecg_output = nullptr;

// Create separate tensor arenas for each model
constexpr int kTensorArenaSize = 32768;
uint8_t ppg_tensor_arena[kTensorArenaSize];
uint8_t ecg_tensor_arena[kTensorArenaSize];

// Clinical thresholds for ECG metrics
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

// RL agent state variables
int battery_level = 10;         // Start with full battery (0-10)
int current_time_step = 0;      // Current time step
bool arr_flag = false;          // Arrhythmia detected flag
bool bp_flag = false;           // Abnormal blood pressure flag
bool fever_flag = false;        // Fever detected flag
int current_action = 7;         // Default action: all sensors on

// Sensor state variables
bool ecg_enabled = true;        // ECG sensor state
bool ppg_enabled = true;        // PPG sensor state
bool temp_enabled = true;       // Temperature sensor state

// Battery simulation
const unsigned long BATTERY_UPDATE_INTERVAL = 5000;  // 1 minute
unsigned long last_battery_update = 0;
const float BATTERY_DRAIN_RATE = 0.1;                // Base drain rate per update

// Thresholds for R-peak detection
const float PEAK_THRESHOLD = 0.7;  // Normalized threshold
const int MIN_RR_SAMPLES = 108;    // Minimum 300ms at 360Hz

// Function declarations for PPG processing
uint32_t calculate_moving_average(uint32_t new_value) {
    uint32_t sum = 0;
    moving_avg_buffer[moving_avg_index] = new_value;
    moving_avg_index = (moving_avg_index + 1) % MOVING_AVG_SIZE;
    
    for (int i = 0; i < MOVING_AVG_SIZE; i++) {
        sum += moving_avg_buffer[i];
    }
    
    return sum / MOVING_AVG_SIZE;
}

float normalize_ppg(uint32_t value) {
    // Update min/max values
    if (value < ppg_min) ppg_min = value;
    if (value > ppg_max) ppg_max = value;
    
    // Prevent division by zero
    if (ppg_max == ppg_min) return 0.5f;
    
    // Normalize to 0-1 range
    return (float)(value - ppg_min) / (float)(ppg_max - ppg_min);
}

// Function declarations for ECG processing
void normalize_ecg() {
    float mean = 0.0;
    float std = 0.0;
    
    // Calculate mean
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        mean += ecg_buffer[i];
    }
    mean /= ECG_SIGNAL_SIZE;
    
    // Calculate standard deviation
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        std += (ecg_buffer[i] - mean) * (ecg_buffer[i] - mean);
    }
    std = sqrt(std / ECG_SIGNAL_SIZE);
    
    // Normalize the data
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        ecg_buffer[i] = (ecg_buffer[i] - mean) / (std + 1e-6);
    }
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
    float filtered[ECG_SIGNAL_SIZE];
    float derivative[ECG_SIGNAL_SIZE];
    float squared[ECG_SIGNAL_SIZE];
    
    // High-pass filter
    for (int i = 1; i < ECG_SIGNAL_SIZE; i++) {
        filtered[i] = HIGH_PASS_ALPHA * (filtered[i-1] + signal[i] - signal[i-1]);
    }
    
    // Low-pass filter
    for (int i = 1; i < ECG_SIGNAL_SIZE; i++) {
        filtered[i] = filtered[i] + LOW_PASS_ALPHA * (filtered[i-1] - filtered[i]);
    }
    
    // Derivative
    for (int i = 2; i < ECG_SIGNAL_SIZE-2; i++) {
        derivative[i] = (2*filtered[i+2] + filtered[i+1] - filtered[i-1] - 2*filtered[i-2]) / 8.0;
    }
    
    // Square
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        squared[i] = derivative[i] * derivative[i];
    }
    
    // Find QRS onset and offset using adaptive threshold
    float max_energy = 0;
    for (int i = peak_index-20; i <= peak_index+20 && i < ECG_SIGNAL_SIZE; i++) {
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
    for (int i = peak_index; i < ECG_SIGNAL_SIZE && i < peak_index + 40; i++) {
        if (squared[i] < threshold) {
            offset = i;
            break;
        }
    }
    
    // Convert samples to milliseconds
    return (offset - onset) * (1000.0 / ECG_SAMPLING_FREQ);
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
        if (idx >= 0 && idx < ECG_SIGNAL_SIZE) {
            pr_baseline += signal[idx];
            pr_samples++;
        }
    }
    
    // Calculate ST level
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        int idx = peak_index + ST_SEGMENT_START + i;
        if (idx >= 0 && idx < ECG_SIGNAL_SIZE) {
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
    Serial.print(metrics.valid_hr ? "true" : "false");
    Serial.print(",\"valid_qrs\":");
    Serial.print(metrics.valid_qrs ? "true" : "false");
    Serial.print(",\"valid_st\":");
    Serial.print(metrics.valid_st ? "true" : "false");
    Serial.print(",\"confidence_score\":");
    Serial.print(metrics.confidence_score);
    Serial.println("}}");
}

// Function to find the best action from Q-table for current state
int getBestAction() {
    int best_action = 2;  // Default: keep PPG on as safety
    float best_q_value = -999999.0;
    bool found_match = false;
    
    // Ensure time_step stays within Q-table range (0-59)
    int lookup_time_step = current_time_step % 60;
    
    // Convert boolean flags to integers for Q-table lookup
    int arr_flag_int = arr_flag ? 1 : 0;
    int bp_flag_int = bp_flag ? 1 : 0;
    int fever_flag_int = fever_flag ? 1 : 0;
    
    Serial.print("Looking for state: Battery=");
    Serial.print(battery_level);
    Serial.print(", TimeStep=");
    Serial.print(lookup_time_step);
    Serial.print(", Flags=");
    Serial.print(arr_flag_int);
    Serial.print(bp_flag_int);
    Serial.println(fever_flag_int);
    
    // STEP 1: First try to find an exact match for all parameters including time_step
    for (int i = 0; i < QTABLE_SIZE; i++) {
        if (qtable[i].battery_disc == battery_level && 
            qtable[i].time_step == lookup_time_step &&
            qtable[i].arr_flag == arr_flag_int && 
            qtable[i].bp_flag == bp_flag_int && 
            qtable[i].fever_flag == fever_flag_int) {
            
            best_action = qtable[i].action;
            best_q_value = qtable[i].q_value;
            found_match = true;
            Serial.print("Found exact match! Action: ");
            Serial.println(best_action);
            break; // Exact match found, exit loop
        }
    }
    
    // STEP 2: If no exact match, try finding best match for this time step
    if (!found_match) {
        for (int i = 0; i < QTABLE_SIZE; i++) {
            if (qtable[i].time_step == lookup_time_step) {
                // Prioritize matching health flags
                int flag_matches = 0;
                if (qtable[i].arr_flag == arr_flag_int) flag_matches++;
                if (qtable[i].bp_flag == bp_flag_int) flag_matches++;
                if (qtable[i].fever_flag == fever_flag_int) flag_matches++;
                
                // Calculate battery difference (closer is better)
                int batt_diff = abs(qtable[i].battery_disc - battery_level);
                
                // Use a scoring system where flag matches are most important
                float score = flag_matches * 10 - batt_diff + qtable[i].q_value;
                
                if (!found_match || score > best_q_value) {
                    best_action = qtable[i].action;
                    best_q_value = score;
                    found_match = true;
                }
            }
        }
        
        if (found_match) {
            Serial.print("Found best match for time step. Action: ");
            Serial.println(best_action);
        }
    }
    
    // STEP 3: If still no match, find best action for current battery level
    if (!found_match) {
        for (int i = 0; i < QTABLE_SIZE; i++) {
            if (qtable[i].battery_disc == battery_level) {
                if (!found_match || qtable[i].q_value > best_q_value) {
                    best_action = qtable[i].action;
                    best_q_value = qtable[i].q_value;
                    found_match = true;
                }
            }
        }
        
        if (found_match) {
            Serial.print("Found match based on battery level. Action: ");
            Serial.println(best_action);
        }
    }
    
    // STEP 4: Safety check: never turn off all sensors
    if (best_action == 0) {
        best_action = 2; // Default to PPG on
        Serial.println("Safety override: Changed action 0 to 2 (PPG on)");
    }
    
    return best_action;
}
// Function to control sensors based on the action
void applySensorAction(int action) {
    // Decode action (3-bit bitmap for 3 sensors)
    ecg_enabled = (action & 0x04) > 0;  // Bit 2: ECG
    ppg_enabled = (action & 0x02) > 0;  // Bit 1: PPG
    temp_enabled = (action & 0x01) > 0; // Bit 0: Temperature
    
    // Control ECG sensor (AD8232)
    if (ecg_enabled) {
        digitalWrite(ECG_SDN_PIN, HIGH);  // Active LOW - power on
        Serial.println("ECG sensor enabled");
    } else {
        digitalWrite(ECG_SDN_PIN, LOW); // Shutdown mode
        Serial.println("ECG sensor disabled");
    }
    
    // Control PPG sensor (MAX30102)
    if (ppg_enabled) {
        particleSensor.wakeUp();
        Serial.println("PPG sensor enabled");
    } else {
        particleSensor.shutDown();
        Serial.println("PPG sensor disabled");
    }
    
    // Control Temperature sensor (MLX90614)
    if (temp_enabled) {
        // Wake up MLX90614 from sleep mode if needed
        Wire.beginTransmission(MLX90614_I2C_ADDR);
        Wire.write(0x03); // Read flags register
        Wire.endTransmission(false);
        Wire.requestFrom(MLX90614_I2C_ADDR, 2);
        Wire.endTransmission();
        Serial.println("Temperature sensor enabled");
    } else {
        // Put MLX90614 to sleep using proper sequence
        Wire.beginTransmission(MLX90614_I2C_ADDR);
        Wire.write(0x06); // Sleep command register
        Wire.write(0x00); // Sleep mode data
        Wire.write(0x00); // Sleep mode data
        Wire.write(0x00); // Sleep mode data  
        Wire.write(0x00); // Sleep mode data
        Wire.write(0x00); // Sleep mode data
        Wire.write(0x00); // Sleep mode data
        Wire.endTransmission();
        Serial.println("Temperature sensor disabled");
    }
    
    // Log current sensor configuration
    Serial.print("Sensor configuration: ");
    Serial.print(ecg_enabled ? "ECG:ON " : "ECG:OFF ");
    Serial.print(ppg_enabled ? "PPG:ON " : "PPG:OFF ");
    Serial.println(temp_enabled ? "TEMP:ON" : "TEMP:OFF");
}

// Function to update the battery level (simulation)
void updateBattery() {
    unsigned long current_time = millis();
    
    if (current_time - last_battery_update >= BATTERY_UPDATE_INTERVAL) {
        last_battery_update = current_time;
        
        // Calculate drain based on active sensors
        float drain = BATTERY_DRAIN_RATE;
        if (ecg_enabled) drain += 0.15;
        if (ppg_enabled) drain += 0.2;
        if (temp_enabled) drain += 0.1;
        
        // Update battery level
        battery_level = max(0, static_cast<int>(battery_level - ceil(drain)));
        
        Serial.print("Battery level: ");
        Serial.print(battery_level * 10);
        Serial.println("%");
        
        // Increment time step
        current_time_step++;
        
        // Check if we need to adjust sensor configuration based on new state
        int new_action = getBestAction();
        if (new_action != current_action) {
            current_action = new_action;
            applySensorAction(current_action);
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(100);  // Short delay for serial to initialize
    Serial.println("Initializing TinyML Health Monitor with Adaptive Sensing...");
    
    // Initialize I2C for both sensors
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(100000); // Use standard 100kHz I2C speed instead of fast mode
    
    // Scan I2C bus for devices
    Serial.println("Scanning I2C bus...");
    byte error, address;
    int deviceCount = 0;
    
    for(address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        error = Wire.endTransmission();
        
        if (error == 0) {
            Serial.print("I2C device found at address 0x");
            if (address < 16) Serial.print("0");
            Serial.print(address, HEX);
            Serial.print(" (");
            if (address == MAX30105_I2C_ADDR) Serial.print("MAX30105");
            else if (address == MLX90614_I2C_ADDR) Serial.print("MLX90614");
            else Serial.print("unknown device");
            Serial.println(")");
            deviceCount++;
        }
    }
    
    if (deviceCount == 0) {
        Serial.println("No I2C devices found!");
        while(1) {
            digitalWrite(ERROR_LED, HIGH);
            delay(500);
            digitalWrite(ERROR_LED, LOW);
            delay(500);
        }
    }
    
    // Check if MAX30102 is available before proceeding with initialization
    if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {  // Try standard speed instead of fast
        Serial.println("MAX30102 not found! Check wiring.");
        while(1) {
            digitalWrite(ERROR_LED, HIGH);
            delay(200);
            digitalWrite(ERROR_LED, LOW);
            delay(200);
        }
    }
    Serial.println("MAX30102 initialized successfully");
    
    // Initialize MLX90614 with explicit Wire instance
    bool temp_sensor_available = mlx.begin(0x5A, &Wire);
    if (!temp_sensor_available) {
        Serial.println("MLX90614 initialization failed! Temperature monitoring disabled.");
        // Continue without temperature sensor instead of getting stuck
    } else {
        Serial.print("MLX90614 initialized successfully, Emissivity = ");
        Serial.println(mlx.readEmissivity());
    }
    
    // Configure PPG sensor
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeIR(0x0A);
    particleSensor.enableDIETEMPRDY();  // Enable die temperature reading
    
    // Initialize ECG pins
    pinMode(LO_PLUS, INPUT);
    pinMode(LO_MINUS, INPUT);
    pinMode(ECG_OUTPUT_PIN, INPUT);
    pinMode(ECG_SDN_PIN, OUTPUT);
    pinMode(NORMAL_LED, OUTPUT);
    pinMode(ABNORMAL_LED, OUTPUT);
    pinMode(ERROR_LED, OUTPUT);
    
    // Set initial state for ECG shutdown pin
    digitalWrite(ECG_SDN_PIN, LOW);  // Initially enabled (active LOW)
    
    // Test all LEDs
    digitalWrite(NORMAL_LED, HIGH);
    digitalWrite(ABNORMAL_LED, HIGH);
    digitalWrite(ERROR_LED, HIGH);
    delay(500);
    digitalWrite(NORMAL_LED, LOW);
    digitalWrite(ABNORMAL_LED, LOW);
    digitalWrite(ERROR_LED, LOW);
    
    // Initialize the battery simulation
    last_battery_update = millis();
    
    // Get initial action from Q-table and apply it
    current_action = getBestAction();
    applySensorAction(current_action);
    
    Serial.print("Initial sensor configuration - Action: ");
    Serial.println(current_action);
    
    // Initialize TFLite for BP estimation model
    ppg_model = tflite::GetModel(model_data);  // Using BP model data
    if (ppg_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("BP Model version mismatch!");
        return;
    }
    
    ppg_interpreter = new tflite::MicroInterpreter(
        ppg_model, resolver, ppg_tensor_arena, kTensorArenaSize, &micro_error_reporter);
    
    if (ppg_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors for BP model!");
        return;
    }
    
    ppg_input = ppg_interpreter->input(0);
    ppg_output = ppg_interpreter->output(0);
    
    // Initialize TFLite for ECG model
    ecg_model = tflite::GetModel(model_tflite);  // Using ECG model data
    if (ecg_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ECG Model version mismatch!");
        return;
    }
    
    ecg_interpreter = new tflite::MicroInterpreter(
        ecg_model, resolver, ecg_tensor_arena, kTensorArenaSize, &micro_error_reporter);
    
    if (ecg_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors for ECG model!");
        return;
    }
    
    ecg_input = ecg_interpreter->input(0);
    ecg_output = ecg_interpreter->output(0);
    
    // Initialize the buffers
    for (int i = 0; i < MOVING_AVG_SIZE; i++) {
        moving_avg_buffer[i] = 0;
    }
    
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        ecg_buffer[i] = 0.0;
    }
    
    Serial.println("Both models initialized successfully!");
}

void loop() {
    static uint32_t last_ppg_sample = 0;
    static uint32_t last_ecg_sample = 0;
    static uint32_t last_temp_read = 0;
    uint32_t current_time = millis();
    
    updateBattery();
    // Improved temperature reading section
    static bool temp_sensor_available = true;
    if (temp_enabled && temp_sensor_available && current_time - last_temp_read >= 1000) {
        last_temp_read = current_time;
        
        try {
            float ambient_temp = mlx.readAmbientTempC();
            float object_temp = mlx.readObjectTempC();
            
            // Check for valid readings (not NaN and within reasonable range)
            if (!isnan(ambient_temp) && !isnan(object_temp) && 
                ambient_temp > -50 && ambient_temp < 100 && 
                object_temp > -50 && object_temp < 100) {
                
                Serial.print("Ambient Temperature: ");
                Serial.print(ambient_temp, 1);
                Serial.print("°C, Body Temperature: ");
                Serial.print(object_temp, 1);
                Serial.println("°C");
            } else {
                Serial.println("Invalid temperature readings, retrying...");
            }
        } catch (...) {
            // If any exception occurs during reading
            digitalWrite(ERROR_LED, HIGH);
            Serial.println("Error reading temperature sensor!");
            delay(100);
            digitalWrite(ERROR_LED, LOW);
        }
    }
    
    // Sample PPG at its specified frequency if enabled
    if (ppg_enabled && (current_time - last_ppg_sample >= PPG_SAMPLING_PERIOD_MS)) {
        last_ppg_sample = current_time;
        
        // Process PPG
        if (particleSensor.available()) {
            
        }
        uint32_t raw_red = particleSensor.getRed();
            uint32_t filtered_red = calculate_moving_average(raw_red);
            
            if (ppg_buffer_index < PPG_SIGNAL_SIZE) {
                ppg_buffer[ppg_buffer_index++] = normalize_ppg(filtered_red);
                
                // Debug PPG signal
                /*Serial.print("Raw: ");
                Serial.print(raw_red);
                Serial.print(", Filtered: ");
                Serial.print(filtered_red);
                Serial.print(", Normalized: ");
                Serial.println(ppg_buffer[ppg_buffer_index-1], 4);*/
            }
            
            // When PPG buffer is full, perform BP estimation
            if (ppg_buffer_index >= PPG_SIGNAL_SIZE) {
                // Process PPG for BP estimation
                memcpy(ppg_input->data.f, ppg_buffer, PPG_SIGNAL_SIZE * sizeof(float));
                
                if (ppg_interpreter->Invoke() == kTfLiteOk) {
                    // Get predictions and scale to mmHg range
                    float systolic = ppg_output->data.f[0] * 40 + 100;  // Scale to reasonable BP range
                    float diastolic = ppg_output->data.f[1] * 30 + 60;  // Scale to reasonable BP range
                    
                    Serial.println("\n=== Blood Pressure Estimation ===");
                    Serial.print("SBP: ");
                    Serial.print(systolic, 1);
                    Serial.print(" mmHg, DBP: ");
                    Serial.print(diastolic, 1);
                    Serial.println(" mmHg");
                    
                    // Update BP flag if blood pressure is abnormal
                    bool old_bp_flag = bp_flag;
                    bp_flag = (systolic > 140 || systolic < 90 || diastolic > 90 || diastolic < 60);
                    
                    if (bp_flag != old_bp_flag) {
                        Serial.println(bp_flag ? "ALERT: Abnormal blood pressure detected!" : "Blood pressure normalized.");
                        // Re-evaluate sensor configuration if BP status changed
                        int new_action = getBestAction();
                        if (new_action != current_action) {
                            current_action = new_action;
                            applySensorAction(current_action);
                        }
                    }
                } else {
                    Serial.println("Error running BP estimation model");
                }
                
                // Reset PPG buffer and dynamic range
                ppg_buffer_index = 0;
                ppg_min = 50000;
                ppg_max = 0;
            }
    }
    
    // Sample ECG at its specified frequency if enabled
    if (ecg_enabled && (current_time - last_ecg_sample >= ECG_SAMPLING_PERIOD_MS)) {
        last_ecg_sample = current_time;
        
        // Check if ECG leads are connected
        if ((digitalRead(LO_PLUS) == 0) && (digitalRead(LO_MINUS) == 0)) {
            digitalWrite(ERROR_LED, LOW);
            
            // Read ECG using ESP32's ADC
            uint16_t raw_ecg = analogRead(ECG_OUTPUT_PIN);
            float ecg_value = raw_ecg / 4095.0f;  // ESP32 ADC is 12-bit
            
            // Process ECG value
            if (ecg_buffer_index < ECG_SIGNAL_SIZE) {
                ecg_buffer[ecg_buffer_index++] = ecg_value;
                
                // Check for R-peak in real-time
                if (detectRPeak(ecg_value, current_time)) {
                    // R-peak detected, can perform real-time analysis if needed
                }
                
                // Debug ECG signal (uncomment if needed)
                // Serial.print("ECG: ");
                // Serial.println(ecg_value, 3);
            }
            
            // When ECG buffer is full, perform analysis
            if (ecg_buffer_index >= ECG_SIGNAL_SIZE) {
                // Create metrics structure
                ECGMetrics metrics;
                
                // 1. Calculate heart rate
                metrics.heart_rate = calculateHeartRate();
                metrics.valid_hr = (metrics.heart_rate >= MIN_HEART_RATE && metrics.heart_rate <= MAX_HEART_RATE);
                
                // 2. Perform classification
                normalize_ecg();
                
                // Copy the ECG data to the input tensor
                memcpy(ecg_input->data.f, ecg_buffer, ECG_SIGNAL_SIZE * sizeof(float));
                
                // Run inference
                if (ecg_interpreter->Invoke() == kTfLiteOk) {
                    float* results = ecg_output->data.f;
                    
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
                        metrics.rhythm_class = ECG_LABELS[max_index];
                        metrics.confidence_score = max_prob;
                        
                        // Check for arrhythmia (any non-normal rhythm classification)
                        bool old_arr_flag = arr_flag;
                        arr_flag = (max_index != 0);  // Anything other than "Normal" class
                        
                        // 3. Calculate QRS duration
                        // Find the highest peak in the current window
                        int peak_index = 0;
                        float max_value = 0;
                        for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
                            if (ecg_buffer[i] > max_value) {
                                max_value = ecg_buffer[i];
                                peak_index = i;
                            }
                        }
                        metrics.qrs_duration = measureQRSDuration(ecg_buffer, peak_index);
                        metrics.valid_qrs = (metrics.qrs_duration >= MIN_QRS_DURATION && 
                                             metrics.qrs_duration <= MAX_QRS_DURATION);
                        
                        // 4. Measure ST elevation
                        metrics.st_elevation = measureSTElevation(ecg_buffer, peak_index);
                        metrics.valid_st = (metrics.st_elevation >= MIN_ST_ELEVATION && 
                                           metrics.st_elevation <= MAX_ST_ELEVATION);
                        
                        // Update LEDs based on classification
                        if (max_index == 0) { // Normal beat
                            digitalWrite(NORMAL_LED, HIGH);
                            digitalWrite(ABNORMAL_LED, LOW);
                        } else { // Abnormal beat
                            digitalWrite(NORMAL_LED, LOW);
                            digitalWrite(ABNORMAL_LED, HIGH);
                        }
                        
                        // Output JSON formatted data
                        Serial.println("\n=== ECG Analysis ===");
                        formatMetricsJSON(metrics);
                        
                        // If arrhythmia status changed, re-evaluate sensor config
                        if (arr_flag != old_arr_flag) {
                            Serial.println(arr_flag ? "ALERT: Arrhythmia detected!" : "Heart rhythm normalized.");
                            int new_action = getBestAction();
                            if (new_action != current_action) {
                                current_action = new_action;
                                applySensorAction(current_action);
                            }
                        }
                    } else {
                        Serial.println("Low confidence in ECG classification, skipping analysis");
                    }
                } else {
                    Serial.println("Error running ECG classification model");
                }
                
                // Shift the buffer by half window size to maintain overlap for continuous monitoring
                for (int i = 0; i < ECG_SIGNAL_SIZE/2; i++) {
                    ecg_buffer[i] = ecg_buffer[i + ECG_SIGNAL_SIZE/2];
                }
                ecg_buffer_index = ECG_SIGNAL_SIZE/2;
            }
        } else {
            digitalWrite(ERROR_LED, HIGH);
            Serial.println("ECG leads disconnected!");
        }
    }
}