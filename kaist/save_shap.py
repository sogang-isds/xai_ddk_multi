import sys

from common import APP_ROOT
sys.path.append('../')
import os
import shap
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from multi_input_model import DDKWav2VecModel


MODEL_DIR = os.path.join(APP_ROOT, 'checkpoints')


# CustomNet 정의 (concat된 입력을 받아 spec_w2v_x와 char_x로 분리)
class CustomNet(nn.Module):
    def __init__(self, model):
        super(CustomNet, self).__init__()
        # model의 char_layer와 final_layer를 사용
        self.char_layer = model.char_layer
        self.final_layer = model.final_layer
    
    def forward(self, concat_x):
        # 입력 텐서를 spec_w2v_x와 char_x로 분리
        spec_w2v_x = concat_x[:, :128]  # 첫 128차원은 spec_w2v_x로 할당
        char_x = concat_x[:, 128:]      # 나머지는 char_x로 할당
        
        # char_layer 통과
        char_x_pro = self.char_layer(char_x)
        
        # CNN + wav2vec + characteristics concat
        total_x = torch.cat([spec_w2v_x, char_x_pro], dim=-1)
        
        # final_layer 통과
        out = self.final_layer(total_x)
        
        return out


def generate_column_dict(column_names, values):
    value_dict = {}
    for column, value in zip(column_names, values):
        if isinstance(value, float):
            value_dict[column] = round(value, 3)
        else:
            value_dict[column] = value

    return value_dict


if __name__ == "__main__":
    df1 = pd.read_csv(os.path.join(APP_ROOT, 'kaist/logs/test_data_1.csv'), dtype={'task_id': str})
    df2 = pd.read_csv(os.path.join(APP_ROOT, 'kaist/logs/test_data_2.csv'), dtype={'task_id': str})

    scaler = joblib.load(os.path.join(APP_ROOT, "scaler.save"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDKWav2VecModel.load_from_checkpoint(os.path.join(MODEL_DIR, "multi_input_model.ckpt")).to(device).eval()

    # CustomNet 인스턴스화
    custom_model = CustomNet(model)

    # df1과 df2에서 첫 두 컬럼(id, task_id)을 제외한 특성 데이터 추출 및 concat
    df1_features = df1.iloc[:, 2:].values.astype(np.float32)  # df1에서 id, task_id 제외
    df2_features = df2.iloc[:, 2:].values.astype(np.float32)  # df2에서 id, task_id 제외

    background_data_df1 = torch.tensor(df1_features).float().to(device)  # df1 데이터를 텐서로 변환
    background_data_df2 = scaler.transform(df2_features)  # df2는 스케일링 후 텐서로 변환
    background_data_df2 = torch.tensor(background_data_df2).float().to(device)
    background_data = torch.cat([background_data_df1, background_data_df2], dim=1)  # concat된 배경 데이터

    explainer = shap.DeepExplainer(custom_model, background_data)

    column_names = [
        "gender", "intelligibility", "var_f0_semitones", "var_f0_hz", "avg_energy", 
        "var_energy", "max_energy", "ddk_rate", "ddk_average", "ddk_std", 
        "ddk_pause_rate", "ddk_pause_average", "ddk_pause_std"
    ]

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

        # spec_w2v_x와 char_x를 concat하여 입력 준비
        concat_x = torch.cat([spec_w2v_x, char_x], dim=-1)  
        shap_values = explainer.shap_values(concat_x)
        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][0]

        # spec_w2v_x에 대한 shap values는 필요 없음.
        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][128:]

        # SHAP 값을 저장 및 순위 생성
        for class_idx, shap_list in enumerate(shap_values):
            shap_dict = generate_column_dict(column_names, shap_list)

            # 절대값 기준으로 feature 순위 계산
            sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            for rank, (feature, value) in enumerate(sorted_features, start=1):
                shap_dict[f'{feature}_rank'] = rank

            # 결과 저장
            result = {
                'id': id,
                'task_id': task_id,
                'shap_class': class_idx,
            }
            result.update(shap_dict)
            results.append(result)

    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)

    # 결과 DataFrame을 CSV 파일로 저장
    results_df.to_csv(os.path.join(APP_ROOT, 'kaist/logs/test_shap.csv'), index=False)