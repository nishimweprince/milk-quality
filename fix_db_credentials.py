import re

# Read the file
with open('insert_data.py', 'r') as file:
    content = file.read()

# Replace the database connection with environment variables
db_conn_pattern = r"""# Database connection
def get_db_connection\(\):
    return mysql\.connector\.connect\(
        host="localhost",
        user="root",
        password="[^"]*",
        database="milkqualitydb"
    \)"""

db_conn_replacement = """# Database connection
def get_db_connection():
    # Use environment variables for secure credential management
    import os
    
    # Default values for development/demo
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'root')
    db_pass = os.environ.get('DB_PASS', '')  # Password should be set via environment variable
    db_name = os.environ.get('DB_NAME', 'milkqualitydb')
    
    return mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_pass,
        database=db_name
    )"""

# Replace the database connection
content = re.sub(db_conn_pattern, db_conn_replacement, content)

# Update the ArduinoReader.connect method
connect_pattern = r"""    def connect\(self, retries=5, delay=2\):
        # Try to find Arduino ports on macOS
        arduino_ports = glob\.glob\('/dev/cu\.usbmodem\*'\) \+ glob\.glob\('/dev/tty\.usbmodem\*'\)
        
        if arduino_ports:
            self\.port = arduino_ports\[0\]  # Use the first Arduino found
            print\(f"Found Arduino port: \{self\.port\}"\)
            try:
                self\.serial_conn = serial\.Serial\(
                    port=self\.port,
                    baudrate=self\.baudrate,
                    timeout=1
                \)
                print\(f"Successfully connected to Arduino on \{self\.port\}"\)
                self\.is_demo_mode = False
                return True
            except Exception as e:
                print\(f"Failed to connect to Arduino: \{e\}"\)
        
        print\("No Arduino found\. Running in demo mode"\)
        self\.is_demo_mode = True
        return False"""

connect_replacement = """    def connect(self, retries=5, delay=2):
        # Try to find Arduino ports on different operating systems
        # Windows - COM ports
        windows_ports = glob.glob('COM*')
        # macOS - USB ports
        mac_ports = glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/tty.usbmodem*')
        
        arduino_ports = windows_ports + mac_ports
        
        # Prioritize COM3 if available for Windows
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

# Replace the connect method
content = re.sub(connect_pattern, connect_replacement, content)

# Update the ArduinoReader.__init__ method
init_pattern = r"""    def __init__\(self, port='/dev/cu\.usbmodem11101', baudrate=9600\):
        self\.port = port
        self\.baudrate = baudrate
        self\.serial_conn = None
        self\.running = False
        self\.lock = threading\.Lock\(\)
        self\.is_demo_mode = False
        self\.connect\(\)"""

init_replacement = """    def __init__(self, port=None, baudrate=9600):
        self.port = port  # Default to None, will be detected in connect()
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.lock = threading.Lock()
        self.is_demo_mode = False
        self.connect()"""

# Replace the init method
content = re.sub(init_pattern, init_replacement, content)

# Update the socketio.run call
run_pattern = r"""if __name__ == '__main__':
    socketio\.run\(app, host='0\.0\.0\.0', port=5001, debug=True\)"""

run_replacement = """if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False, log_output=True)"""

# Replace the run call
content = re.sub(run_pattern, run_replacement, content)

# Add os import if not present
if 'import os' not in content:
    content = content.replace('import pandas as pd', 'import pandas as pd\nimport os')

# Write the updated content back to the file
with open('insert_data.py', 'w') as file:
    file.write(content)

print("insert_data.py has been updated successfully")
