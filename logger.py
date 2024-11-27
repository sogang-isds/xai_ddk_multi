import logging

from config import DEBUG

logger = logging.getLogger('main')

formatter = logging.Formatter('%(asctime)s (%(module)s:%(lineno)d) %(name)s - %(levelname)s: %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
