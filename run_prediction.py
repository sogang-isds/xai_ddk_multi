from argparse import ArgumentParser
import os
import joblib
import numpy as np
import pandas as pd
import shap
from captum.attr import IntegratedGradients
import torch

from common import APP_ROOT
from inference import prepare_data
from kaist.evaluate_shap import (
    cal_scores,
    get_strengths_and_weaknesses,
    get_majority_key,
    majority_vote_by_id,
)
from kaist.save_data import FeaturesGenerator
from kaist.save_ig import CustomNet
from kaist.save_shap import generate_column_dict
from multi_input_model import DDKWav2VecModel
from utils.utils import save_to_json

def filter_dict(d, keys):
    return {k: v for k, v in d.items() if k in keys}

class ExplainDDK:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DDKWav2VecModel.load_from_checkpoint(model_path).to(self.device).eval()
        self.features_generator = FeaturesGenerator(self.model)

        self.custom_model = CustomNet(self.model)

        self.scaler = joblib.load(os.path.join(APP_ROOT, "scaler.save"))

        background_data = self.load_background_data()

        self.explainer = {}
        self.explainer["shap"] = shap.DeepExplainer(self.custom_model, background_data)
        self.explainer["ig"] = IntegratedGradients(self.custom_model)

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

        # 분석할 피쳐 리스트
        self.target_features = [
            "ddk_rate",
            "ddk_average",
            "ddk_std",
            "ddk_pause_rate",
            "ddk_pause_average",
            "ddk_pause_std",
        ]
        self.idx2sev = {0: "normal", 1: "mild-to-moderate", 2: "severe"}

    def load_background_data(self):
        df1 = pd.read_csv(
            os.path.join(APP_ROOT, "resources/test_data_1.csv"), dtype={"task_id": str}
        )
        df2 = pd.read_csv(
            os.path.join(APP_ROOT, "resources/test_data_2.csv"), dtype={"task_id": str}
        )

        # df1과 df2에서 첫 두 컬럼(id, task_id)을 제외한 특성 데이터 추출 및 concat
        df1_features = df1.iloc[:, 2:].values.astype(
            np.float32
        )  # df1에서 id, task_id 제외
        df2_features = df2.iloc[:, 2:].values.astype(
            np.float32
        )  # df2에서 id, task_id 제외

        background_data_df1 = (
            torch.tensor(df1_features).float().to(self.device)
        )  # df1 데이터를 텐서로 변환
        background_data_df2 = self.scaler.transform(
            df2_features
        )  # df2는 스케일링 후 텐서로 변환
        background_data_df2 = torch.tensor(background_data_df2).float().to(self.device)
        background_data = torch.cat(
            [background_data_df1, background_data_df2], dim=1
        )  # concat된 배경 데이터

        return background_data

    def load_baseline_data(self):
        df1 = pd.read_csv(
            os.path.join(APP_ROOT, "resources/test_data_1.csv"), dtype={"task_id": str}
        )
        df2 = pd.read_csv(
            os.path.join(APP_ROOT, "resources/test_data_2.csv"), dtype={"task_id": str}
        )

        df1_features = df1.iloc[:, 2:].values.astype(np.float32)
        df2_features = self.scaler.transform(df2.iloc[:, 2:].values.astype(np.float32))
        df1_mean = np.mean(df1.iloc[:, 2:].values.astype(np.float32), axis=0)
        df2_mean = np.mean(
            self.scaler.transform(df2.iloc[:, 2:].values.astype(np.float32)), axis=0
        )
        baseline_data_df1 = torch.tensor(df1_mean).float().to(self.device).unsqueeze(0)
        baseline_data_df2 = torch.tensor(df2_mean).float().to(self.device).unsqueeze(0)
        baseline = torch.cat([baseline_data_df1, baseline_data_df2], dim=1)

        return baseline

    def get_feature_dict(self, features_df):
        output_dict = {}
        
        for idx, row in features_df.iterrows():
            task_id = row["task_id"]
            feature_dict = row.to_dict()
            
            filtered_dict = filter_dict(feature_dict, self.target_features)
            
            output_dict[task_id] = filtered_dict

        return output_dict

    def analyze_result(self, df, xai_method="shap"):
        df_labels = pd.read_csv(os.path.join(APP_ROOT, "resources/test_labels.csv"))
        shap_df = pd.read_csv(os.path.join(APP_ROOT, f"resources/test_{xai_method}.csv"))

        # id를 기준으로 severity 정보를 shap_df에 병합
        shap_df = shap_df.merge(df_labels[["id", "severity"]], on="id", how="left")

        # shap_class가 2인 데이터 필터링
        shap_class_2_df = shap_df[shap_df["shap_class"] == 2]

        # normal_mean과 severe_mean 계산
        normal_mean = {}
        severe_mean = {}

        features = self.target_features

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

        output_dict = {}
        for idx, row in shap_df_class_2.iterrows():
            # id = row['id']
            task_id = row["task_id"]
            severity = row["severity"]

            task_dict = {}
            print(task_id)

            sample_df = pd.DataFrame([row], columns=features, index=[0])

            shap_dict = sample_df.to_dict(orient="records")[0]
            print(f"Shap Dict: {shap_dict}")
            task_dict["xai_values"] = shap_dict

            # sample_df = df[features].iloc[[idx]]
            feature_score, _ = cal_scores(sample_df, normal_mean, severe_mean)
            print(f"Feature Score: {feature_score}")
            task_dict["feature_score"] = feature_score

            # Strength와 Weakness 구하기
            feature_area = get_strengths_and_weaknesses(feature_score)
            print(feature_area)
            task_dict["feature_area"] = feature_area

            # 우수 영역과 개선 영역을 예측값으로 변환
            prediction = {}
            for feature in features:
                if feature in feature_area["strength"]:
                    prediction[feature] = 0  # strength는 우수(0)
                elif feature in feature_area["weakness"]:
                    prediction[feature] = 2  # weakness는 개선(2)
                else:
                    prediction[feature] = 1  # 나머지는 보통(1)

            # 예측된 값 저장
            predicted_classes.append([id, severity, prediction])

            output_dict[task_id] = task_dict

        # 예측값 DataFrame으로 변환
        predicted_df = pd.DataFrame(
            predicted_classes, columns=["id", "severity", "predictions"]
        )

        for feature in features:
            predicted_df[f"{feature}_predicted"] = predicted_df["predictions"].apply(
                lambda x: x[feature]
            )

        # 다수결로 계산된 id별 피처별 예측값 DataFrame 생성
        majority_df = majority_vote_by_id(predicted_df, features)

        # df to dict
        df_dict = majority_df.to_dict(orient="records")[0]

        final_severity = get_majority_key(predicted_df["severity"].values)
        print(f"Final Severity: {final_severity}")

        # id 칼럼을 삭제
        del df_dict["id"]
        print(df_dict)
        output_dict["summary"] = df_dict

        return output_dict, final_severity

    def predict_ddk(self, audio_filepath, gender, task_id):
        audio = prepare_data(audio_filepath).to(self.device)
        mel_x = self.model.mel_spectrogram(audio.unsqueeze(0))
        mel_x = self.model.db_converter(mel_x)

        spec_w2v_x, raw_features = self.features_generator.get_features(
            audio_filepath, gender
        )

        spec_w2v_x = torch.tensor(spec_w2v_x).float().to(self.device)
        spec_w2v_x = spec_w2v_x.unsqueeze(dim=0)

        scaled_features = self.scaler.transform([raw_features])
        scaled_features = torch.tensor(scaled_features).float().to(self.device)

        concat_x = torch.cat([spec_w2v_x, scaled_features], dim=-1)

        out = self.model(mel_x, audio, scaled_features)
        severity = torch.argmax(out, dim=-1)[0]
        severity = severity.item()  # tensor to int

        filtered_features = generate_column_dict(
            self.column_names, raw_features, round_digits=3
        )

        # self.column_names와 raw_features를 합혀서 dictionary로 변환
        feature_dict = {
            "task_id": task_id,
        }
        feature_dict.update(filtered_features)
        print(f"[Inference Result] {severity} ({self.idx2sev[int(severity)]})")

        return severity, feature_dict, concat_x

    def explain_ddk_ig(self, severity, concat_x, task_id):
        #
        # IG values 계산
        #
        baseline = self.load_baseline_data()
        explainer = self.explainer["ig"]

        results = []

        # 각 클래스(0, 1, 2)에 대해 Integrated Gradients 계산
        for class_idx in [0, 1, 2]:
            attributions, _ = explainer.attribute(
                concat_x,
                target=class_idx,
                return_convergence_delta=True,
                baselines=baseline,
            )
            # 첫 128개 값은 제거 후 저장
            ig_values = attributions.squeeze().cpu().detach().numpy()[128:]

            # 결과 저장
            result = {
                # 'id': id,
                "task_id": task_id,
                "severity": severity,
                "shap_class": class_idx,
            }

            value_dict = generate_column_dict(
                self.column_names, ig_values, round_digits=5
            )
            result.update(value_dict)
            results.append(result)

        return results

    def explain_ddk_shap(self, severity, concat_x, task_id):
        #
        # SHAP values 계산
        #
        explainer = self.explainer["shap"]
        shap_values = explainer.shap_values(concat_x)

        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][0]

        # spec_w2v_x에 대한 shap values는 필요 없음.
        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][128:]

        results = []

        # SHAP 값을 저장 및 순위 생성
        for class_idx, shap_list in enumerate(shap_values):
            shap_dict = generate_column_dict(
                self.column_names, shap_list, round_digits=5
            )

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
                "severity": severity,
                "shap_class": class_idx,
            }
            result.update(shap_dict)
            results.append(result)

        return results

    def exaplain_ddks(self, audio_files, gender: int):
        result_shap_list = []
        result_ig_list = []
        features = []
        for task_id, audio_file in audio_files:
            severity, feature_dict, concat_x = self.predict_ddk(
                audio_file, gender, task_id
            )
            # SHAP values 계산
            result_shap = self.explain_ddk_shap(severity, concat_x, task_id)
            result_shap_list.extend(result_shap)

            # IG values 계산
            result_ig = self.explain_ddk_ig(severity, concat_x, task_id)
            result_ig_list.extend(result_ig)

            # feature 저장
            features.append(feature_dict)

        result_shap_df = pd.DataFrame(result_shap_list)
        result_ig_df = pd.DataFrame(result_ig_list)
        features_df = pd.DataFrame(features)

        shap_dict, final_severity = self.analyze_result(result_shap_df, xai_method='shap')

        ig_dict, _ = self.analyze_result(result_ig_df, xai_method='ig')

        feature_dict = self.get_feature_dict(features_df)

        output_dict = {}
        output_dict["features"] = feature_dict
        output_dict["shap"] = shap_dict
        output_dict["ig"] = ig_dict
        output_dict["severity"] = final_severity
        output_dict["gender"] = gender

        return output_dict


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

    result_dict = explain_ddk.exaplain_ddks(files, gender)
    shap_result = result_dict["shap"]
    ig_result = result_dict["ig"]

    # save result to json
    save_to_json(result_dict, "result_all.json")
    save_to_json(shap_result, "result_shap.json")
    save_to_json(ig_result, "result_ig.json")
