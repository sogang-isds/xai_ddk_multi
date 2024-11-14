import sys

from common import APP_ROOT
sys.path.append('../')
import os
import glob
import pandas as pd
from math import isnan
from argparse import Namespace

import torch
import torchaudio

from multi_input_model import DDKWav2VecModel
from intelligibility_model import PTKModel
from vad_model import VADModel
from get_features import (
    prepare_data, my_padding, get_speech_interval,
    get_intelligibility, get_phonation_features, get_prosody_respiration_features
)
from ddk.features import DDK


MODEL_DIR = os.path.join(APP_ROOT, 'checkpoints')


def prepare_audio(wav):
    """
    First time setup of the dataset
    """

    y, sr = torchaudio.load(wav)  # type:ignore
    
    if sr != 16000:
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=16000)
    
    y = y.mean(0)
    
    y = my_padding(y, 16000*15)

    return y.unsqueeze(0)


class FeaturesGenerator(object):
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
        self.ddk_path = os.path.join(APP_ROOT, 'ddk/')

        intel_args = {
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 100,
            'n_gpu': 1,
        }

        intel_args = Namespace(**intel_args)
        
        intel_path = os.path.join(MODEL_DIR, "intelligibility_model.ckpt")
        self.intel_model = PTKModel.load_from_checkpoint(intel_path,n_classes=5,args=intel_args)
        self.intel_model.to(self.device)
        self.intel_model.eval()

        vad_args = {
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 100,
            'n_gpu': 1,
            'hidden_size': 128,
            'num_layers': 16
        }
        
        vad_args = Namespace(**vad_args)
        vad_path = os.path.join(MODEL_DIR, "vad_model.ckpt")
        self.vad_model = VADModel.load_from_checkpoint(vad_path, args=vad_args)
        self.vad_model.to(self.device)

        self.model = model

    def get_intelligibility(self, path):       
        y = prepare_data(path)

        y = my_padding(y, 16000 * 15).to(self.device)
        y = self.intel_model.mel_converter(y)
        y = y.unsqueeze(0)
        y = y.unsqueeze(0)
        y = self.intel_model(y)
        
        intelligibility = torch.argmax(y, dim=-1).cpu().numpy()
        
        return int(intelligibility[0])       

    def get_phonation_features(self, path):        
        ddk_features = DDK(path, self.ddk_path)
        ddk_features = [0 if isnan(x) else x for x in ddk_features]
        return ddk_features

    def get_prosody_respiration_features(self, path, threshold, min_dur):
        audio_tensor, sample_rate  = torchaudio.load(path)
        if sample_rate != 16000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000
        audio_tensor = audio_tensor.mean(0)
        
        dur = len(audio_tensor) / sample_rate
        
        pad_size = int(16000*0.75)
        padding = torch.zeros((pad_size))
        
        audio_tensor = torch.concat([padding, audio_tensor])
        
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        mel = self.vad_model.spec_converter(audio_tensor)
        
        out = self.vad_model(mel)
        out = self.vad_model.sigmoid(out)

        out = (out > threshold).float().cpu()

        out = get_speech_interval(out, min_dur)
        
        ddk_rate, ddk_avg, ddk_std, ddk_pause_rate, pause_avg, ddk_pause_std = cal_features_from_speech_interval(out, dur)
        
        return ddk_rate, ddk_avg, ddk_std, ddk_pause_rate, pause_avg, ddk_pause_std

    def get_features(self, audio_path, gender):
        intelligibility = get_intelligibility(audio_path)
        phonation_features = get_phonation_features(audio_path)

        threshold = 0.7
        min_dur = 0.08
        prosody_respiration_features = get_prosody_respiration_features(audio_path, threshold, min_dur)
        
        features = [gender, intelligibility]
        features += phonation_features
        features += prosody_respiration_features

        audio = prepare_audio(audio_path).to(self.device)
        mel_x = self.model.mel_spectrogram(audio.unsqueeze(0))
        mel_x = self.model.db_converter(mel_x)

        spec_x = mel_x
        w2v_x = audio
        char_x = features

        # CNN(ResNet)
        spec_x, _ = self.model.resnet_model(spec_x)
        
        # CNN projection
        spec_x = self.model.post_spec_layer(spec_x)
        spec_x = self.model.relu(spec_x)
        spec_x = self.model.post_spec_dropout(spec_x)

        spec_attn_x = spec_x.reshape(spec_x.shape[0], 1, -1)
        
        # wav2vec 2.0 
        w2v_x = self.model.wav2vec(w2v_x)[0]
        w2v_x = torch.matmul(spec_attn_x, w2v_x)
        w2v_x = w2v_x.reshape(w2v_x.shape[0], -1)
        
        # wav2vec projection
        w2v_x = self.model.post_w2v_layer(w2v_x)
        w2v_x = self.model.relu(w2v_x)
        w2v_x = self.model.post_wv2_droput(w2v_x)
        
        # CNN + wav2vec concat and projection
        spec_w2v_x = torch.cat([spec_x, w2v_x], dim=-1)
        spec_w2v_x = self.model.post_attn_layer(spec_w2v_x)
        spec_w2v_x = self.model.relu(spec_w2v_x)
        spec_w2v_x = self.model.post_attn_dropout(spec_w2v_x)

        spec_w2v_x = spec_w2v_x.squeeze()  # (128,) 모양으로 차원 제거
        spec_w2v_x = spec_w2v_x.tolist()  # 리스트로 변환

        return spec_w2v_x, features 


if __name__ == "__main__":
    df_labels = pd.read_csv(os.path.join(APP_ROOT, 'kaist/logs/test_labels.csv'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDKWav2VecModel.load_from_checkpoint(os.path.join(MODEL_DIR, "multi_input_model.ckpt")).to(device).eval()

    features_generator = FeaturesGenerator(model)

    column_names = [
        "id",  
        "task_id", 
        # ==================== #
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
        "ddk_pause_std"
    ]

    audio_paths = glob.glob(f'{APP_ROOT}/kaist/data/*/*/*/*.wav')

    # Initialize an empty list to hold all features data
    data_1 = []
    data_2 = []

    for i, audio_path in enumerate(audio_paths):
        print(i, len(audio_paths))
        filename = os.path.basename(audio_path)[:-4]
        if 'nia' in filename:
            id = filename.split('_')[0] + '_' + filename.split('_')[1]
            task_id = filename.split('_')[6]
            gender = filename.split('_')[3]
        else:
            id = filename.split('_')[0]    
            task_id = filename.split('_')[1]
            gender = filename.split('_')[2]

        if id in df_labels['id'].values:
            spec_w2v_x, features = features_generator.get_features(audio_path, gender)
            data_1.append([id] + [task_id] + spec_w2v_x)
            data_2.append([id] + [task_id] + features)
        else:
            print(f"{id} is NOT present in df_labels['id']")

    # spec_w2v_x에 대한 컬럼 이름을 자동으로 생성 (spec_w2v_x_1, spec_w2v_x_2, ..., spec_w2v_x_128)
    spec_w2v_columns = [f"spec_w2v_x_{i+1}" for i in range(128)]

    # Create a DataFrame with the collected features for df_1
    df_1 = pd.DataFrame(data_1, columns=["id", "task_id"] + spec_w2v_columns)

    # Save the DataFrame to a CSV file
    df_1.to_csv(os.path.join(APP_ROOT, 'kaist/logs/test_data_1.csv'), index=False)

    # Create a DataFrame with the collected features
    df_2 = pd.DataFrame(data_2, columns=column_names)

    # Save the DataFrame to a CSV file
    df_2.to_csv(os.path.join(APP_ROOT, 'kaist/logs/test_data_2.csv'), index=False)