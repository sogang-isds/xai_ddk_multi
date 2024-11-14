
import os
import pandas as pd

from common import APP_ROOT





if __name__ == "__main__":
    # CSV 파일을 읽는 코드
    df = pd.read_csv(os.path.join(APP_ROOT, 'kaist/data/DDK 언어치료사 자문 리스트.csv'))

    # /퍼/, /터/, /커/, /퍼터커/ 컬럼을 drop
    df = df.drop(['/퍼/', '/터/', '/커/', '/퍼터커/'], axis=1)

    # 컬럼 이름 변경
    df = df.rename(columns={
        '기존 심각도': 'severity', 
        '화자번호': 'id',
        '초당 음절 발음 횟수': 'ddk_rate',
        '음절 발음 시간의 평균': 'ddk_average',
        '음절 발음 시간의 표준편차': 'ddk_std',
        '초당 쉼 횟수': 'ddk_pause_rate',
        '쉼 시간의 평균': 'ddk_pause_average',
        '쉼 시간의 표준편차': 'ddk_pause_std'
    })

    # 수정된 데이터프레임을 test_labels.csv로 저장
    df.to_csv(os.path.join(APP_ROOT, 'kaist/logs/test_labels.csv'), index=False)

    # 수정된 데이터프레임 출력
    print(df.head())