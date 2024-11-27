import argparse
import os
import tempfile

import requests

from celery.utils.log import get_task_logger

from common import APP_ROOT
# from common import MODEL_DIR
from config import *
from run_prediction import ExplainDDK
from utils.ddk_utils import convert_gender, convert_age

logger = get_task_logger(__name__)

from celery import Celery

if DEBUG:
    logger.setLevel('DEBUG')

VERIFY_SSL = False if DEBUG else True

app = Celery(CELERY_TASK_NAME, broker=MQ_CELERY_BROKER_URL, backend=MQ_CELERY_BACKEND_URL)

explain_ddk = ExplainDDK(
        model_path=os.path.join(APP_ROOT, "checkpoints/multi_input_model.ckpt")
    )


@app.task(bind=True)
def recognize_ddk_multi(self, file_paths, gender):
    # shorten the task_id
    task_id = self.request.id
    task_id = task_id[:8]


    try:
        t = tempfile.TemporaryDirectory()
        output_path = t.name

        files = []
        for task_id, file_path in file_paths:
            # HTTP URL로 넘어온 경우 파일 다운로드
            if file_path.startswith("http"):
                res = requests.get(file_path, allow_redirects=True, verify=VERIFY_SSL)
                file = os.path.basename(file_path)

                file_path = os.path.join(output_path, file)

                with open(file_path, "wb") as f:
                    f.write(res.content)

            files.append((task_id, file_path))

        result_dict = explain_ddk.exaplain_ddks(files, gender)

    except Exception as e:
        template = "An exception of type {0} occured. Arguments:\n{1!r}"
        debug_msg = template.format(type(e).__name__, e.args)
        logger.error(f'({task_id}) {debug_msg}')

        return False

    return result_dict


if __name__ == "__main__":
    # set worker options
    worker_options = {
        'loglevel': 'INFO',
        'traceback': True,
        'concurrency': 1,
        'pool': 'solo'
    }

    # set queue
    app.conf.task_default_queue = CELERY_TASK_NAME

    worker = app.Worker(**worker_options)

    # start worker
    worker.start()
