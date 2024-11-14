import sys

from common import APP_ROOT
sys.path.append('../')
import os
import joblib
import pandas as pd
import numpy as np

import torch

from multi_input_model import DDKWav2VecModel



MODEL_DIR = os.path.join(APP_ROOT, 'checkpoints')


if __name__ == "__main__":
    scaler = joblib.load(os.path.join(APP_ROOT, "scaler.save"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDKWav2VecModel.load_from_checkpoint(os.path.join(MODEL_DIR, "multi_input_model.ckpt")).to(device).eval()

    df1 = pd.read_csv(os.path.join(APP_ROOT, 'kaist/logs/test_data_1.csv'), dtype={'task_id': str})
    df2 = pd.read_csv(os.path.join(APP_ROOT, 'kaist/logs/test_data_2.csv'), dtype={'task_id': str})

    results = []

    for (idx1, row1), (idx2, row2) in zip(df1.iterrows(), df2.iterrows()):
        id = row1['id']
        task_id = row1['task_id']

        # 첫 두 컬럼(id, task_id)를 제외하고 나머지를 선택
        row1_filtered = row1.iloc[2:]  # iloc[2:]를 사용하여 2번째 이후의 값들만 선택
        
        # 나머지 값을 PyTorch 텐서로 변환
        spec_w2v_x = torch.tensor(row1_filtered.values.astype(np.float32), dtype=torch.float32, device=device)
        spec_w2v_x = spec_w2v_x.unsqueeze(dim=0)

        # 첫 두 컬럼(id, task_id)를 제외하고 나머지를 선택
        row2_filtered = row2.iloc[2:]  # iloc[2:]를 사용하여 2번째 이후의 값들만 선택
        char_x = scaler.transform([row2_filtered.values])
        char_x = torch.tensor(char_x).float().to(device)
        
        # Characteristics projection
        char_x_pro = model.char_layer(char_x)

        # CNN + wav2vec + characteristics concat
        total_x = torch.cat([spec_w2v_x, char_x_pro], dim=-1)
        
        # Final output
        out = model.final_layer(total_x)

        severity = torch.argmax(out, dim=-1)[0]
        severity = int(severity)

        # 결과 저장
        results.append({'id': id, 'task_id': task_id, 'severity': severity})

    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)

    # 결과 DataFrame을 CSV 파일로 저장
    results_df.to_csv(os.path.join(APP_ROOT, 'kaist/logs/test_preds.csv'), index=False)