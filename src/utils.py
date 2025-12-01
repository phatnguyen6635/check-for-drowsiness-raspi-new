import cv2
from pathlib import Path
import yaml
from yaml import Loader
from pathlib import Path
import time
import serial

OPEN_CMD_HEX = "A00101A2"   # Command to turn relay ON
CLOSE_CMD_HEX = "A00100A1"  # Command to turn relay OFF

def load_config(config_path='configs/configs.yaml'):
    """Load configuration from YAML file with error handling"""
    try:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as stream:
            config = yaml.load(stream, Loader=Loader)
        
        required_keys = ['model_path', 'num_faces', 'min_face_detection_confidence',
                         'min_face_presence_confidence', 'blink_threshold_pitch',
                         'blink_threshold_wo_pitch', 'frame_width',
                         'frame_height', 'frame_rate']
        
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")
        
        return config
    
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

def initialize_serial(port="/dev/ttyUSB0", baudrate=9600, timeout=1, logger=None):
    """
    Initialize the USB-to-Serial connection.

    Returns:
        serial.Serial object if successful, otherwise None.
    """

    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.1)  # Allow the serial port to settle
        if logger:
            logger.info(f"Serial opened on {port} at {baudrate} baud")
        return ser
    except Exception as e:
        if logger:
            logger.error(f"Failed to open serial port {port}: {e}")
        return None

def close_serial(ser, logger=None):
    """Safely close the serial port if it is open."""
    if ser is None:
        return
    try:
        ser.close()
        if logger:
            logger.info("Serial connection closed")
    except Exception as e:
        if logger:
            logger.error(f"Error closing serial: {e}")

def _send_hex_bytes(ser, hexstr, logger=None):
    """
    Convert a hex string into bytes and send it to the relay.

    Args:
        ser: serial port object
        hexstr: hex string without spaces (e.g., 'A00101A2')

    Returns:
        True on success, False on failure.
    """
    if ser is None:
        if logger:
            logger.warning("Serial connection is None. Cannot send command.")
        return False

    try:
        data = bytes.fromhex(hexstr)
        ser.write(data)
        ser.flush()
        if logger:
            logger.info(f"Sent serial command: {hexstr}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Serial send error: {e}")
        return False
    
def set_relay(logger, state, serial_conn=None):
    """
    Control the relay or fallback to GPIO if serial is not available.

    Args:
        gpio_enabled: whether GPIO mode is enabled
        led_pin: GPIO pin number (for fallback)
        logger: logger object
        state: True = ON, False = OFF
        serial_conn: serial port object (if relay uses USB serial)

    Priority:
        1. If serial_conn is provided → send relay command via USB serial.
        2. Otherwise fallback to GPIO output.
    """

    if serial_conn:
        cmd = OPEN_CMD_HEX if state else CLOSE_CMD_HEX
        success = _send_hex_bytes(serial_conn, cmd, logger)
        if not success:
            logger.error("Failed to set relay state through serial.")
        return
    return
  
def cleanup_resources(cam, detector, serial_conn, logger):
    """Cleanup all resources safely"""
    logger.info("Cleaning up resources...")
    
    try:
        cam.close()
    except Exception as e:
        logger.error(f"Error closing camera: {e}")
    
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Error destroying windows: {e}")
    
    try:
        detector.close()
    except Exception as e:
        logger.error(f"Error closing detector: {e}")
    
    if serial_conn:
        close_serial(serial_conn, logger)
