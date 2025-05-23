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
import os
from ml_processor import ml_processor
import psycopg2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'milkquality'
socketio = SocketIO(app, async_mode='threading')

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

def get_db_connection():
    import os
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_pass = os.environ.get('DB_PASS', 'nishimwe')  # Password should be set via environment variable
    db_name = os.environ.get('DB_NAME', 'milkquality')
    db_port = os.environ.get('DB_PORT', '5433')
    return psycopg2.connect(
        host=db_host,
        user=db_user,
        password=db_pass,
        dbname=db_name,
        port=db_port
    )

db_connection = get_db_connection()

def save_to_csv(data):
    """Save readings to CSV as backup"""
    df = pd.DataFrame([data])
    df.to_csv('milk_readings.csv', mode='a', header=not os.path.exists('milk_readings.csv'), index=False)
    print(f"Data saved to CSV: {data}")

class ArduinoReader:
    def __init__(self, port=None, baudrate=9600):
        self.port = port  # Default to None, will be detected in connect()
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.lock = threading.Lock()
        self.is_demo_mode = False
        self.connect()

    def connect(self, retries=5, delay=2):
        windows_ports = glob.glob('COM*')
        mac_ports = glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/tty.usbmodem*')
        
        arduino_ports = windows_ports + mac_ports
        
        if 'COM3' in arduino_ports:
            self.port = 'COM3'
            print(f"Found Arduino on preferred port: {self.port}")
        elif arduino_ports:
            self.port = arduino_ports[0]  # Use the first Arduino found
            print(f"Found Arduino port: {self.port}")
        
        if arduino_ports:
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
        """Generate demo data with more realistic variations across quality categories."""
        scenario = random.choices(
            ['safe', 'monitor', 'mild_concern', 'serious_concern', 'critical'],
            weights=[0.35, 0.25, 0.2, 0.15, 0.05],  # Adjusted weights for realistic distribution
            k=1
        )[0]
        
        ph_base = 0
        ec_base = 0
        protein_base = 0
        turbidity_base = 0
        scc_base = 0
        
        if scenario == 'safe':
            ph_base = 6.7
            ec_base = 500
            protein_base = 3.3
            turbidity_base = 2
            scc_base = 120000
        elif scenario == 'monitor':
            ph_base = 6.4
            ec_base = 600
            protein_base = 3.1
            turbidity_base = 5
            scc_base = 300000
        elif scenario == 'mild_concern':
            ph_base = 6.1
            ec_base = 700
            protein_base = 2.9
            turbidity_base = 9
            scc_base = 800000
        elif scenario == 'serious_concern':
            ph_base = 5.8
            ec_base = 900
            protein_base = 2.7
            turbidity_base = 15
            scc_base = 3000000
        else:  # critical
            ph_base = 5.5
            ec_base = 1100
            protein_base = 2.4
            turbidity_base = 25
            scc_base = 7000000
        
        ph = round(random.uniform(ph_base - 0.3, ph_base + 0.3), 2)
        ec = int(random.uniform(ec_base * 0.8, ec_base * 1.2))
        protein = round(random.uniform(protein_base * 0.9, protein_base * 1.1), 2)
        turbidity = int(random.uniform(turbidity_base * 0.7, turbidity_base * 1.3))
        scc = int(random.uniform(scc_base * 0.6, scc_base * 1.4))  # Larger variation for SCC
        
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
                line = self.serial_conn.readline().decode('utf-8').strip()
                print(f"Raw data received: '{line}'")
                
                if not line or 'DEBUG:' in line or '---' in line:
                    print("Skipping debug or empty line")
                    return None
                    
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
                    ml_prediction = ml_processor.predict_quality(real_data)
                    
                    combined_data = {**real_data, **ml_prediction}
                    
                    global latest_readings
                    latest_readings = combined_data
                    
                    save_to_db(combined_data)
                    
                    save_to_csv(combined_data)
                    
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
        cursor.execute("""
            INSERT INTO milk_sample (sample_time, milk_quality, action_needed) 
            VALUES (%s, %s, %s)
        """, (
            datetime.now(timezone(timedelta(hours=2))),
            data.get('milk_quality', 'Unknown'),
            data.get('action_needed', 'Unknown')
        ))
        sample_id = cursor.fetchone()[0] if cursor.description else None
        if sample_id is None:
            cursor.execute('SELECT currval(pg_get_serial_sequence(\'milk_sample\', \'sample_id\'))')
            sample_id = cursor.fetchone()[0]
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

@app.route('/')
def index():
    return render_template('index.html')

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
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False, log_output=True)
