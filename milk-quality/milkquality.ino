/*
 * Milk Quality Sensor System
 * 
 * This sketch reads data from multiple sensors to analyze milk quality:
 * - pH Sensor: Connected to A0
 * - EC Sensor: Connected to A1
 * - Protein Sensor: Connected to A2
 * - Turbidity Sensor: Connected to A3
 * 
 * Data is sent via Serial to a Python application for ML-based analysis.
 */

#include <Wire.h>

// Sensor Pins (no conflicts)
#define PH_SENSOR A0
#define EC_SENSOR A1
#define PROTEIN_SENSOR A2
#define TURBIDITY_SENSOR A3

// Alert Pins
#define BUZZER 8
#define LED_GREEN 9
#define LED_YELLOW 10
#define LED_RED 11

// Calibration Factors
const float pH_OFFSET = 0.0;
const float EC_OFFSET = 0.0;
const float EC_SCALE = 1.5;
const float TURBIDITY_FACTOR_A = 2.0;
const float TURBIDITY_FACTOR_B = 2.0;
const float PROTEIN_FACTOR_W1 = 0.8;
const float PROTEIN_FACTOR_W2 = 0.5;
const float PROTEIN_CONSTANT = 0.2;
const float SCC_FACTOR = 1.2;
const float SCC_CONSTANT = 1000.0;

// Thresholds for CMT Grading
const int SCC_THRESHOLDS[] = {200000, 400000, 1200000, 5000000};

// Timing variables
unsigned long previousMillis = 0;
const long interval = 5000;  // Send data every 5 seconds

void setup() {
    Serial.begin(9600);
    while (!Serial); // Wait for serial port to connect
    
    // Clear any garbage in serial buffer
    while(Serial.available() > 0) Serial.read();
    
    pinMode(BUZZER, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    
    Serial.println("Milk Quality Sensor System Initialized");
    delay(2000); // Allow sensors to stabilize
}

void loop() {
    unsigned long currentMillis = millis();
    
    // Check if it's time to send data
    if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis;
        
        // Read sensors
        float ph = readPH();
        float ec = readEC();
        float protein = calculateProtein(ec, ph);
        float turbidity = readTurbidity();
        float scc = calculateSCC(ec, turbidity);
        String quality = classifyMilk(scc);

        // Create JSON string
        String jsonData = "{";
        jsonData += "\"pH\":" + String(ph, 2) + ",";
        jsonData += "\"turbidity\":" + String(turbidity, 2) + ",";
        jsonData += "\"ec\":" + String(ec, 2) + ",";
        jsonData += "\"protein\":" + String(protein, 2) + ",";
        jsonData += "\"scc\":" + String(scc, 0);
        jsonData += "}";
        
        // Send JSON data
        Serial.println(jsonData);
        
        // Debug output
        debugOutput(ph, ec, protein, turbidity, scc, quality);
        
        // Visual and audio alerts
        updateAlerts(quality);
    }
}

void debugOutput(float ph, float ec, float protein, float turbidity, float scc, String quality) {
    Serial.print("DEBUG: pH="); Serial.print(ph);
    Serial.print(", EC="); Serial.print(ec);
    Serial.print(", Protein="); Serial.print(protein);
    Serial.print(", Turbidity="); Serial.print(turbidity);
    Serial.print(", SCC="); Serial.print(scc);
    Serial.print(", Quality="); Serial.println(quality);
    Serial.println("------------------------");
}

void updateAlerts(String quality) {
    if (quality == "Negative") {
        digitalWrite(LED_GREEN, HIGH);
        digitalWrite(LED_YELLOW, LOW);
        digitalWrite(LED_RED, LOW);
        noTone(BUZZER);
    } 
    else if (quality == "Trace" || quality == "Weak +") {
        digitalWrite(LED_GREEN, LOW);
        digitalWrite(LED_YELLOW, HIGH);
        digitalWrite(LED_RED, LOW);
        tone(BUZZER, 500, 500);
    } 
    else {
        digitalWrite(LED_GREEN, LOW);
        digitalWrite(LED_YELLOW, LOW);
        digitalWrite(LED_RED, HIGH);
        tone(BUZZER, 1000, 1000);
    }
}

float readPH() {
    int raw = analogRead(PH_SENSOR);
    float voltage = raw * (5.0 / 1023.0);
    return (3.5 * voltage) + pH_OFFSET;
}

float readEC() {
    int raw = analogRead(EC_SENSOR);
    float voltage = raw * (5.0 / 1023.0);
    return (voltage - EC_OFFSET) * EC_SCALE;
}

float readTurbidity() {
    int raw = analogRead(TURBIDITY_SENSOR);
    float voltage = raw * (5.0 / 1023.0);
    return TURBIDITY_FACTOR_A * voltage + TURBIDITY_FACTOR_B;
}

float calculateProtein(float ec, float ph) {
    return (PROTEIN_FACTOR_W1 * ec) + (PROTEIN_FACTOR_W2 * ph) + PROTEIN_CONSTANT;
}

float calculateSCC(float ec, float turbidity) {
    return (SCC_FACTOR * ec) + (2.0 * turbidity) + SCC_CONSTANT;
}

String classifyMilk(float scc) {
    if (scc < SCC_THRESHOLDS[0]) return "Negative";
    else if (scc < SCC_THRESHOLDS[1]) return "Trace";
    else if (scc < SCC_THRESHOLDS[2]) return "Weak +";
    else if (scc < SCC_THRESHOLDS[3]) return "Distinct +";
    return "Definite +";
}