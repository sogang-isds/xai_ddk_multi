from argparse import ArgumentParser
import os

import joblib
import numpy as np
import pandas as pd
import shap
import torch

from common import APP_ROOT
from inference import prepare_data
from kaist.evaluate_shap import (
    cal_scores,
    get_strengths_and_weaknesses,
    majority_vote_by_id,
)
from kaist.save_data import FeaturesGenerator
from kaist.save_ig import CustomNet
from kaist.save_shap import generate_column_dict
from multi_input_model import DDKWav2VecModel


class ExplainDDK:
    def __init__(self, model_path):

        self.model = DDKWav2VecModel.load_from_checkpoint(model_path).to(device).eval()
        self.features_generator = FeaturesGenerator(self.model)

        custom_model = CustomNet(self.model)

        self.scaler = joblib.load(os.path.join(APP_ROOT, "scaler.save"))
        
        background_data = self.load_background_data()
        self.explainer = shap.DeepExplainer(custom_model, background_data)

        self.column_names = [
            "gender",
            "intelligibility",
            "var_f0_semitones",
            "var_f0_hz",
            "avg_energy",
            "var_energy",
            "max_energy",
            "ddk_rate",
            "ddk_average",
            "ddk_std",
            "ddk_pause_rate",
            "ddk_pause_average",
            "ddk_pause_std",
        ]

    def load_background_data(self):
        df1 = pd.read_csv(
            os.path.join(APP_ROOT, "kaist/logs/test_data_1.csv"), dtype={"task_id": str}
        )
        df2 = pd.read_csv(
            os.path.join(APP_ROOT, "kaist/logs/test_data_2.csv"), dtype={"task_id": str}
        )

        

        # df1과 df2에서 첫 두 컬럼(id, task_id)을 제외한 특성 데이터 추출 및 concat
        df1_features = df1.iloc[:, 2:].values.astype(
            np.float32
        )  # df1에서 id, task_id 제외
        df2_features = df2.iloc[:, 2:].values.astype(
            np.float32
        )  # df2에서 id, task_id 제외

        background_data_df1 = (
            torch.tensor(df1_features).float().to(device)
        )  # df1 데이터를 텐서로 변환
        background_data_df2 = self.scaler.transform(
            df2_features
        )  # df2는 스케일링 후 텐서로 변환
        background_data_df2 = torch.tensor(background_data_df2).float().to(device)
        background_data = torch.cat(
            [background_data_df1, background_data_df2], dim=1
        )  # concat된 배경 데이터

        return background_data

    def analyze_shap(self, df):
        df_labels = pd.read_csv(os.path.join(APP_ROOT, "kaist/logs/test_labels.csv"))
        shap_df = pd.read_csv(os.path.join(APP_ROOT, "kaist/logs/test_shap.csv"))

        # id를 기준으로 severity 정보를 shap_df에 병합
        shap_df = shap_df.merge(df_labels[["id", "severity"]], on="id", how="left")

        # shap_class가 2인 데이터 필터링
        shap_class_2_df = shap_df[shap_df["shap_class"] == 2]

        # normal_mean과 severe_mean 계산
        normal_mean = {}
        severe_mean = {}

        # 분석할 피쳐 리스트
        features = [
            "ddk_rate",
            "ddk_average",
            "ddk_std",
            "ddk_pause_rate",
            "ddk_pause_average",
            "ddk_pause_std",
        ]

        for feature in features:
            # severity가 0인 경우의 shap_class 2 saliency 평균 (Normal)
            normal_mean[feature] = shap_class_2_df[shap_class_2_df["severity"] == 0][
                feature
            ].mean()

            # severity가 2인 경우의 shap_class 2 saliency 평균 (Severe)
            severe_mean[feature] = shap_class_2_df[shap_class_2_df["severity"] == 2][
                feature
            ].mean()

        # shap_df_class_2 = shap_df[shap_df['shap_class'] == 2]
        shap_df_class_2 = df[df["shap_class"] == 2]

        # 결과 저장 리스트
        predicted_classes = []

        # 각 샘플에 대해 strength와 weakness를 예측
        id = "test"
        for idx, row in shap_df_class_2.iterrows():
            # id = row['id']
            # task_id = row['task_id']

            sample_df = pd.DataFrame([row], columns=df.columns, index=[0])
            # sample_df = df[features].iloc[[idx]]
            feature_score, _ = cal_scores(sample_df, normal_mean, severe_mean)

            print(feature_score)

            # Strength와 Weakness 구하기
            feature_area = get_strengths_and_weaknesses(feature_score)

            # 우수 영역과 개선 영역을 예측값으로 변환
            prediction = {}
            for feature in df.columns:
                if feature in feature_area["strength"]:
                    prediction[feature] = 0  # strength는 우수(0)
                elif feature in feature_area["weakness"]:
                    prediction[feature] = 2  # weakness는 개선(2)
                else:
                    prediction[feature] = 1  # 나머지는 보통(1)

            # 예측된 값 저장
            predicted_classes.append([id, prediction])

        # print(predicted_classes)

        # 예측값 DataFrame으로 변환
        predicted_df = pd.DataFrame(predicted_classes, columns=["id", "predictions"])

        for feature in features:
            predicted_df[f"{feature}_predicted"] = predicted_df["predictions"].apply(
                lambda x: x[feature]
            )

        # 다수결로 계산된 id별 피처별 예측값 DataFrame 생성
        majority_df = majority_vote_by_id(predicted_df, features)

        # df to dict
        df_dict = majority_df.to_dict(orient="records")
        print(df_dict)

        return df_dict

    def explain_ddk(self, audio_filepath, gender, task_id):
        audio = prepare_data(audio_filepath).to(device)
        mel_x = self.model.mel_spectrogram(audio.unsqueeze(0))
        mel_x = self.model.db_converter(mel_x)

        spec_w2v_x, features = self.features_generator.get_features(
            audio_filepath, gender
        )

        spec_w2v_x = torch.tensor(spec_w2v_x).float().to(device)
        spec_w2v_x = spec_w2v_x.unsqueeze(dim=0)

        features = self.scaler.transform([features])
        features = torch.tensor(features).float().to(device)

        concat_x = torch.cat([spec_w2v_x, features], dim=-1)

        shap_values = self.explainer.shap_values(concat_x)

        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][0]

        # spec_w2v_x에 대한 shap values는 필요 없음.
        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][128:]

        results = []

        # SHAP 값을 저장 및 순위 생성
        for class_idx, shap_list in enumerate(shap_values):
            shap_dict = generate_column_dict(self.column_names, shap_list)

            # 절대값 기준으로 feature 순위 계산
            sorted_features = sorted(
                shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )
            for rank, (feature, value) in enumerate(sorted_features, start=1):
                shap_dict[f"{feature}_rank"] = rank

            # 결과 저장
            result = {
                # 'id': id,
                "task_id": task_id,
                "shap_class": class_idx,
            }
            result.update(shap_dict)
            results.append(result)

        out = self.model(mel_x, audio, features)
        severity = torch.argmax(out, dim=-1)[0]

        print(f"[Inference Result] {severity} ({idx2sev[int(severity)]})")

        return results

    def exaplain_ddks(self, audio_files, gender):
        results = []
        for task_id, audio_file in audio_files:
            result = self.explain_ddk(audio_file, gender, task_id)
            results.extend(result)

        results_df = pd.DataFrame(results)

        self.analyze_shap(results_df)

        return results


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "--audio_filepath",
        type=str,
        default="./nia_HC0002_50_0_1_5_002_1.wav",
        help="절대경로로 입력",
    )
    args.add_argument("--gender", type=str, default="F", help="F or M")

    files = [
        ("002", os.path.join(APP_ROOT, "sample_data/nia_HC0033_50_0_2_1_002_1.wav")),
        ("003", os.path.join(APP_ROOT, "sample_data/nia_HC0033_50_0_2_1_003_1.wav")),
        ("004", os.path.join(APP_ROOT, "sample_data/nia_HC0033_50_0_2_1_004_1.wav")),
        ("005", os.path.join(APP_ROOT, "sample_data/nia_HC0033_50_0_2_1_005_1.wav")),
    ]

    args = args.parse_args()

    idx2sev = {0: "normal", 1: "mild-to-moderate", 2: "severe"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_filepath = args.audio_filepath

    # relative path to absolute path
    audio_filepath = os.path.abspath(audio_filepath)

    # check file exists
    if not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"File not found: {audio_filepath}")

    gender = 0 if args.gender == "M" else 1

    explain_ddk = ExplainDDK(
        model_path=os.path.join(APP_ROOT, "checkpoints/multi_input_model.ckpt")
    )
    # explain_ddk.load_shap_data()

    # severity = explain_ddk.explain_ddk(audio_filepath, gender)
    results = explain_ddk.exaplain_ddks(files, gender)

    # print(f"[Inference Result] {severity} ({idx2sev[int(severity)]})")
