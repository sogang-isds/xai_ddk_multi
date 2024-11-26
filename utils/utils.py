import json


def load_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_to_json(data, filename='data.json'):
    if filename[-4:] != 'json':
        filename += '.json'

    with open(f'{filename}', 'w', encoding='utf-8') as fw:
        json.dump(data, fw, indent=4, ensure_ascii=False)