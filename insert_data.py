from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import serial
import time
import threading
import mysql.connector
from datetime import datetime, timezone, timedelta
import atexit
import random
import glob
import json
import pandas as pd
from ml_processor import ml_processor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'milkquality'
socketio = SocketIO(app, async_mode='threading')

# Global variables to store latest readings
latest_readings = {
    'ph': 0,
    'turbidity': 0,
    'ec': 0,
    'protein': 0,
    'scc': 0,
    'milk_quality': 'Unknown',
    'action_needed': 'Unknown',
    'timestamp': ''
}

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Companyyacu@00",
        database="milkqualitydb"
    )

db_connection = get_db_connection()

def save_to_csv(data):
    """Save readings to CSV as backup"""
    df = pd.DataFrame([data])
    df.to_csv('milk_readings.csv', mode='a', header=not pd.io.common.file_exists('milk_readings.csv'), index=False)
    print(f"Data saved to CSV: {data}")

class ArduinoReader:
    def __init__(self, port='/dev/cu.usbmodem11101', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.lock = threading.Lock()
        self.is_demo_mode = False
        self.connect()

    def connect(self, retries=5, delay=2):
        # Try to find Arduino ports on macOS
        arduino_ports = glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/tty.usbmodem*')
        
        if arduino_ports:
            self.port = arduino_ports[0]  # Use the first Arduino found
            print(f"Found Arduino port: {self.port}")
            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=1
                )
                print(f"Successfully connected to Arduino on {self.port}")
                self.is_demo_mode = False
                return True
            except Exception as e:
                print(f"Failed to connect to Arduino: {e}")
        
        print("No Arduino found. Running in demo mode")
        self.is_demo_mode = True
        return False

    def generate_demo_data(self):
        """Generate demo data for all possible milk quality cases."""
        scenario = random.choices(
            ['safe', 'monitor', 'not_safe'],
            weights=[0.5, 0.3, 0.2],  # Adjust weights as desired
            k=1
        )[0]

        if scenario == 'safe':
            # Negative: all values in normal range
            scc = random.randint(50000, 180000)
            ph = round(random.uniform(6.6, 6.9), 2)
            ec = random.randint(450, 550)
            protein = round(random.uniform(3.2, 3.5), 2)
            turbidity = random.randint(1, 3)
        elif scenario == 'monitor':
            # Trace or Weak +: slightly elevated SCC or other mild abnormality
            scc = random.randint(200001, 1200000)
            ph = round(random.uniform(6.2, 6.5), 2)
            ec = random.randint(550, 700)
            protein = round(random.uniform(3.0, 3.3), 2)
            turbidity = random.randint(3, 8)
        else:
            # Not safe: Distinct + or Definite +, high SCC or other severe abnormality
            scc = random.randint(1200001, 6000000)
            ph = round(random.uniform(5.5, 6.1), 2)
            ec = random.randint(700, 1200)
            protein = round(random.uniform(2.5, 3.0), 2)
            turbidity = random.randint(8, 30)

        data = {
            'ph': ph,
            'ec': ec,
            'protein': protein,
            'turbidity': turbidity,
            'scc': scc,
            'timestamp': datetime.now(timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S'),
            'is_demo': True
        }
        return data

    def read_data(self):
        if self.is_demo_mode:
            return self.generate_demo_data()
            
        with self.lock:
            if not self.serial_conn or not self.serial_conn.is_open:
                return None
            try:
                # Read and print raw data for debugging
                line = self.serial_conn.readline().decode('utf-8').strip()
                print(f"Raw data received: '{line}'")
                
                # Skip empty lines and debug messages
                if not line or 'DEBUG:' in line or '---' in line:
                    print("Skipping debug or empty line")
                    return None
                    
                # Only process lines that look like CSV data
                if ',' in line:
                    print(f"Processing CSV data: {line}")
                    try:
                        values = [float(x.strip()) for x in line.split(',') if x.strip()]
                        if len(values) == 5:  # pH, EC, Protein, Turbidity, SCC
                            data = {
                                'ph': round(values[0], 2),
                                'ec': int(values[1]),
                                'protein': round(values[2], 2),
                                'turbidity': int(values[3]),
                                'scc': int(values[4]),
                                'timestamp': datetime.now(timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')
                            }
                            print(f"Successfully parsed data: {data}")
                            return data
                        else:
                            print(f"Wrong number of values: expected 5, got {len(values)}")
                    except ValueError as ve:
                        print(f"Error parsing values: {ve}")
                        print(f"Values attempted to parse: {line.split(',')}")
                else:
                    print("Line is not CSV format, skipping")
                    
            except Exception as e:
                print(f"Error reading from Arduino: {e}")
                self.connect()
            return None

    def start_reading(self):
        self.running = True
        def read_loop():
            while self.running:
                real_data = self.read_data()
                if real_data:
                    # Get ML prediction
                    ml_prediction = ml_processor.predict_quality(real_data)
                    
                    # Combine sensor data with ML prediction
                    combined_data = {**real_data, **ml_prediction}
                    
                    # Update latest readings
                    global latest_readings
                    latest_readings = combined_data
                    
                    # Save to database
                    save_to_db(combined_data)
                    
                    # Save to CSV as backup
                    save_to_csv(combined_data)
                    
                    # Emit to websocket
                    socketio.emit('sensor_update', combined_data)
                time.sleep(5)  # Wait for 5 seconds before next reading
        threading.Thread(target=read_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

def save_to_db(data):
    cursor = db_connection.cursor()
    try:
        # Insert into milk_sample table
        cursor.execute("""
            INSERT INTO milk_sample (sample_time, milk_quality, action_needed) 
            VALUES (%s, %s, %s)
        """, (
            datetime.now(timezone(timedelta(hours=2))),
            data.get('milk_quality', 'Unknown'),
            data.get('action_needed', 'Unknown')
        ))
        sample_id = cursor.lastrowid
        
        # Insert sensor readings
        cursor.execute("INSERT INTO ph_sensor (sample_id, ph_value) VALUES (%s, %s)", 
                      (sample_id, data['ph']))
        cursor.execute("INSERT INTO ec_sensor (sample_id, ec_value) VALUES (%s, %s)", 
                      (sample_id, data['ec']))
        cursor.execute("INSERT INTO protein_sensor (sample_id, protein_value) VALUES (%s, %s)", 
                      (sample_id, data['protein']))
        cursor.execute("INSERT INTO turbidity_sensor (sample_id, turbidity_value) VALUES (%s, %s)", 
                      (sample_id, data['turbidity']))
        cursor.execute("INSERT INTO scc_sensor (sample_id, scc_value) VALUES (%s, %s)", 
                      (sample_id, data['scc']))
        
        db_connection.commit()
    except Exception as e:
        print(f"Database error: {e}")
        db_connection.rollback()
    finally:
        cursor.close()

arduino_reader = ArduinoReader()

# Web routes
@app.route('/')
def index():
    return render_template('index.html')

# API endpoints
@app.route('/api/latest', methods=['GET'])
def get_latest():
    """API endpoint to get latest readings"""
    return jsonify(latest_readings)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions from manual input"""
    data = request.json
    sensor_data = {
        'ph': data.get('pH', 0),
        'turbidity': data.get('turbidity', 0),
        'ec': data.get('ec', 0),
        'protein': data.get('protein', 0),
        'scc': data.get('scc', 0)
    }
    
    prediction = ml_processor.predict_quality(sensor_data)
    return jsonify(prediction)

@app.route('/api/history', methods=['GET'])
def get_history():
    """API endpoint to get historical data"""
    try:
        df = pd.read_csv('milk_readings.csv')
        return jsonify(df.tail(100).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    if not arduino_reader.running:
        arduino_reader.start_reading()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def cleanup():
    arduino_reader.stop()
    db_connection.close()

atexit.register(cleanup)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)