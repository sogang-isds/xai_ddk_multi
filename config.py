import logging
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()

if dotenv_path:
    load_dotenv(dotenv_path)
else:
    raise FileNotFoundError('No .env file found')

def str_to_bool(value):
    return value.lower() in ('true', 't', 'yes', 'y', '1')

# DEBUG
DEBUG = str_to_bool(os.getenv('DEBUG', 'false'))

# RabbitMQ Info
MQ_HOST = os.getenv('MQ_HOST')
MQ_USER_ID = os.getenv('MQ_USER_ID')
MQ_USER_PW = os.getenv('MQ_USER_PW')
MQ_PORT = os.getenv('MQ_PORT')

#
# Celery Info
#
MQ_CELERY_BROKER_URL = f'amqp://{MQ_USER_ID}:{MQ_USER_PW}@{MQ_HOST}:{MQ_PORT}//'
MQ_CELERY_BACKEND_URL = f'rpc://{MQ_USER_ID}:{MQ_USER_PW}@{MQ_HOST}:{MQ_PORT}//'

# Celery Queue Name
CELERY_TASK_NAME = os.getenv('CELERY_TASK_NAME')