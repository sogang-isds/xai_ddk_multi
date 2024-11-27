def convert_age(age):
    # 1: 10~20대, 3: 30~40대, 5: 50~60대, 7: 70대 이상
    if age < 30:
        return 1
    elif age < 50:
        return 3
    elif age < 70:
        return 5
    else:
        return 7


def convert_gender(gender):
    if gender == 0:
        return "M"
    else:
        return "F"
