# 애플리케이션 루트 디렉토리
import os


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, 'checkpoints')