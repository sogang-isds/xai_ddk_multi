import sys

from common import APP_ROOT
sys.path.append('../')
import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error




def cal_scores(df, normal_mean, severe_mean):
    ascending = [feature for feature in normal_mean if normal_mean[feature] < severe_mean[feature]]
    descending = [feature for feature in normal_mean if normal_mean[feature] >= severe_mean[feature]]

    prosody = ['ddk_rate', 'ddk_average', 'ddk_std']
    respiration = ['ddk_pause_rate', 'ddk_pause_average', 'ddk_pause_std']
    vocalization = []

    scores = {}

    # Ascending 방향 피쳐 계산
    for a in ascending:
        a_max = severe_mean[a]
        if df.loc[0, a] < normal_mean[a]:
            scores[a] = 100
        elif df.loc[0, a] > a_max:
            scores[a] = 0
        else:
            score = (df.loc[0, a] - normal_mean[a]) / (a_max - normal_mean[a])
            scores[a] = round((1 - score) * 100, 2)

    # Descending 방향 피쳐 계산
    for d in descending:
        d_min = severe_mean[d]
        if df.loc[0, d] > normal_mean[d]:
            scores[d] = 100
        elif df.loc[0, d] < d_min:
            scores[d] = 0
        else:
            score = (df.loc[0, d] - d_min) / (normal_mean[d] - d_min)
            scores[d] = round(score * 100, 2)

    # Subsystem Scores
    prosody_scores = np.mean([scores[p] for p in prosody if p in scores])
    respiration_scores = np.mean([scores[r] for r in respiration if r in scores])
    vocalization_scores = 0 if not vocalization else np.mean([scores[v] for v in vocalization if v in scores])

    subsystem_score = {'prosody': prosody_scores, 'respiration': respiration_scores, 'vocalization': vocalization_scores}

    return scores, subsystem_score


def get_strengths_and_weaknesses(feature_score):
    # 강점과 약점 저장할 리스트 초기화
    strengths, weaknesses = [], []

    # 분석할 주요 피쳐 리스트
    features = ['ddk_rate', 'ddk_average', 'ddk_std', 'ddk_pause_rate', 'ddk_pause_average', 'ddk_pause_std']

    # feature_score의 절대값을 기준으로 중요도 순으로 정렬
    sorted_features = sorted(features, key=lambda x: abs(feature_score[x]), reverse=True)

    # 강점 및 약점 구분
    for feature in sorted_features:
        score = feature_score.get(feature, None)
        if score is None:
            continue
        if score >= 70:
            strengths.append(feature)
        elif score <= 30:
            weaknesses.append(feature)

    feature_area = {'strength': strengths, 'weakness': weaknesses}
    return feature_area


def majority_vote_by_id(predicted_df, features):
    id_predictions = []
    
    for id_val, group in predicted_df.groupby('id'):
        id_prediction = {'id': id_val}
        # 각 피처에 대해 다수결로 최종 예측값 계산
        for feature in features:
            # 동일한 id에 대해 각 task_id에서의 예측값을 모아서 다수결 계산
            feature_predictions = group[f'{feature}_predicted'].values
            majority_vote = Counter(feature_predictions).most_common(1)[0][0]
            id_prediction[f'{feature}_majority'] = majority_vote
        
        id_predictions.append(id_prediction)
    
    return pd.DataFrame(id_predictions)


if __name__ == "__main__":
    # 데이터 불러오기
    df_labels = pd.read_csv(os.path.join(APP_ROOT, 'kaist/logs/test_labels.csv'))
    shap_df = pd.read_csv(os.path.join(APP_ROOT, 'kaist/logs/test_ig.csv'))

    # 분석할 피쳐 리스트
    features = ['ddk_rate', 'ddk_average', 'ddk_std', 'ddk_pause_rate', 'ddk_pause_average', 'ddk_pause_std']

    # id를 기준으로 severity 정보를 shap_df에 병합
    shap_df = shap_df.merge(df_labels[['id', 'severity']], on='id', how='left')

    # shap_class가 2인 데이터 필터링
    shap_class_2_df = shap_df[shap_df['shap_class'] == 2]

    # normal_mean과 severe_mean 계산
    normal_mean = {}
    severe_mean = {}

    for feature in features:
        # severity가 0인 경우의 shap_class 2 saliency 평균 (Normal)
        normal_mean[feature] = shap_class_2_df[shap_class_2_df['severity'] == 0][feature].mean()
        
        # severity가 2인 경우의 shap_class 2 saliency 평균 (Severe)
        severe_mean[feature] = shap_class_2_df[shap_class_2_df['severity'] == 2][feature].mean()

    shap_df_class_2 = shap_df[shap_df['shap_class'] == 2]
    df = shap_df_class_2
    labels_df = df_labels

    # 결과 저장 리스트
    predicted_classes = []

    # 각 샘플에 대해 strength와 weakness를 예측
    for idx, row in df.iterrows():
        id = row['id']
        task_id = row['task_id']
        
        sample_df = pd.DataFrame([row], columns=df.columns, index=[0])
        # sample_df = df[features].iloc[[idx]]
        feature_score, _ = cal_scores(sample_df, normal_mean, severe_mean)
        
        # Strength와 Weakness 구하기
        feature_area = get_strengths_and_weaknesses(feature_score)
        
        # 우수 영역과 개선 영역을 예측값으로 변환
        prediction = {}
        for feature in df.columns:
            if feature in feature_area['strength']:
                prediction[feature] = 0  # strength는 우수(0)
            elif feature in feature_area['weakness']:
                prediction[feature] = 2  # weakness는 개선(2)
            else:
                prediction[feature] = 1  # 나머지는 보통(1)
        
        # 예측된 값 저장
        predicted_classes.append([id, task_id, prediction])

    # 예측값 DataFrame으로 변환
    predicted_df = pd.DataFrame(predicted_classes, columns=['id', 'task_id', 'predictions'])

    # 각 피처별 예측값을 분리하여 DataFrame에 추가
    features = ['ddk_rate', 'ddk_average', 'ddk_std', 'ddk_pause_rate', 'ddk_pause_average', 'ddk_pause_std']
    for feature in features:
        predicted_df[f'{feature}_predicted'] = predicted_df['predictions'].apply(lambda x: x[feature])

    # test_labels.csv와 predicted_df를 id와 task_id를 기준으로 병합하여 비교
    comparison_df = labels_df.merge(predicted_df[['id', 'task_id'] + [f'{feature}_predicted' for feature in features]], on=['id'])

    # 정확도 계산
    accuracy = {}
    for feature in features:
        comparison_df[f'{feature}_correct'] = comparison_df[feature] == comparison_df[f'{feature}_predicted']
        accuracy[feature] = comparison_df[f'{feature}_correct'].mean()

    # 정확도 출력
    print("Feature-wise Accuracy compared to test_labels.csv:")
    for feature, acc in accuracy.items():
        print(f"Accuracy for {feature}: {acc:.2f}")

    # 다수결로 계산된 id별 피처별 예측값 DataFrame 생성
    majority_df = majority_vote_by_id(predicted_df, features)

    # labels_df에서 각 id별 실제 라벨과 비교 (여기서는 id별로 병합, 동일한 라벨 사용 가정)
    comparison_df = labels_df.groupby('id').first().reset_index()  # 각 id별로 첫 번째 row만 남김
    comparison_df = comparison_df.merge(majority_df, on='id')

    # 피처별 정확도 계산
    accuracy = {}
    for feature in features:
        comparison_df[f'{feature}_correct'] = comparison_df[feature] == comparison_df[f'{feature}_majority']
        accuracy[feature] = comparison_df[f'{feature}_correct'].mean()

    # 피처별 정확도 출력
    print('\n')
    print("Accuracy by Feature after Majority Voting by ID:")
    for feature, acc in accuracy.items():
        print(f"Accuracy for {feature}: {acc:.2f}")

    # 피처별로 MSE 계산
    mse = {}
    for feature in features:
        mse[feature] = mean_squared_error(comparison_df[feature], comparison_df[f'{feature}_majority'])

    # MSE 출력
    print('\n')
    print("Mean Squared Error (MSE):")
    for feature, error in mse.items():
        print(f"MSE for {feature}: {error:.2f}")

    comparison_df.to_csv(os.path.join(APP_ROOT, 'kaist/logs/evaluate_ig.csv'), index=False)