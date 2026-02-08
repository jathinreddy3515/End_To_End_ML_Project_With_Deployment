



import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
##for testing the logger
#print('Log file path:', LOG_FILE_PATH)
#print('File exists before basicConfig:', os.path.exists(LOG_FILE_PATH))

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
##for testing the logger
#print('File exists after basicConfig:', os.path.exists(LOG_FILE_PATH))

#logging.info('Test message')
#print('File exists after logging.info:', os.path.exists(LOG_FILE_PATH))
#print('Files in logs dir:', os.listdir(logs_path))