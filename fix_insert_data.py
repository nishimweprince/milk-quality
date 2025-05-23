import os

with open('insert_data.py', 'r') as file:
    content = file.read()

if 'import os' not in content:
    content = content.replace('import pandas as pd', 'import pandas as pd\nimport os')

old_connect = """    def connect(self, retries=5, delay=2):
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
        return False"""

new_connect = """    def connect(self, retries=5, delay=2):
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
        return False"""

content = content.replace(old_connect, new_connect)

old_init = """    def __init__(self, port='/dev/cu.usbmodem11101', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.lock = threading.Lock()
        self.is_demo_mode = False
        self.connect()"""

new_init = """    def __init__(self, port=None, baudrate=9600):
        self.port = port  # Default to None, will be detected in connect()
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.lock = threading.Lock()
        self.is_demo_mode = False
        self.connect()"""

content = content.replace(old_init, new_init)

old_run = "    socketio.run(app, host='0.0.0.0', port=5001, debug=True)"
new_run = "    socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False, log_output=True)"
content = content.replace(old_run, new_run)

with open('insert_data.py', 'w') as file:
    file.write(content)

print("insert_data.py has been updated successfully")
