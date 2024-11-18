from common import APP_ROOT
from intelligibility_model import PTKModel
from vad_model import VADModel
from ddk.features import DDK
# from loguru import logging
import torch
import torchaudio

import numpy as np
from math import ceil, isnan
from argparse import Namespace
import os

def prepare_data(wav):
    """
    First time setup of the dataset
    """

    y, sr = torchaudio.load(wav)  # type:ignore
    
    if sr != 16000:
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=16000)
    
    y = y.mean(0)
    if len(y.shape) != 1:
        y = y.squeeze(0)
        
    pad_size = int(16000*0.75)
    padding = torch.zeros((pad_size))
    
    y = torch.concat([padding, y])

    return y

def my_padding(y, pad_size):
    if pad_size != int(pad_size):
        pad_size = int(pad_size)

    n_repeat = ceil(pad_size / len(y))
    rep = y.repeat(n_repeat)
    return torch.nn.functional.pad(rep, (0, pad_size - len(rep)))


def time_to_mel_frame(time, sample_rate=16000):
    hop_length = int(sample_rate * 0.0025)
    return int(time * sample_rate / hop_length)

def frame_to_time(frame, sample_rate=16000):
    hop_length = int(sample_rate * 0.0025)
    return frame * hop_length / sample_rate

def get_speech_interval(output, min_dur=0.07):
    min_frame = time_to_mel_frame(min_dur)
    sil_min_frame = time_to_mel_frame(0.015)
    triggered = False
    sil_triggered = False
    start = 0
    sil_start = 0
    
    for i, out in enumerate(output):
        if out == 0:
            if not sil_triggered:
                sil_triggered = True
                sil_start = i

        else:
            if sil_triggered:
                sil_triggered = False
                sil_end = i-1
                sil_dur = sil_end - sil_start + 1
                if sil_dur < sil_min_frame:
                    output[sil_start: sil_end+1] = 1

    for i, out in enumerate(output):
        if out == 0:
            if not triggered:
                continue
                
            triggered=False
            end = i -1
            speech_dur = end - start + 1
            if speech_dur < min_frame:
                output[start:end+1] = 0
        else:
            if not triggered:
                triggered = True
                start = i
                
    return output

def cal_features_from_speech_interval(output,dur):
    pause_rate = 0
    pause = []
    speech = []
    
    triggered = False
    sil_triggered = False
    start = 0
    sil_start = 0

    min_sil_dur = 0.14
    
    for i, out in enumerate(output):
        if out == 0:
            if not sil_triggered:
                sil_triggered = True
                sil_start = i
            if triggered:
                triggered=False
                end = i -1
                speech_dur = frame_to_time(end - start + 1)
                speech.append(speech_dur)
        else:
            if not triggered:
                triggered = True
                start = i  
            if sil_triggered:
                sil_triggered = False
                sil_end = i-1
                sil_dur = frame_to_time(sil_end - sil_start + 1)
                if sil_dur > min_sil_dur:
                    pause_rate += 1
                pause.append(sil_dur)
 
    return float(len(speech)) / dur, 1000*np.mean(speech), 1000*np.std(speech), float(pause_rate)/dur, 1000*np.mean(pause), 1000*np.std(pause)

def get_prosody_respiration_features(path, threshold, min_dur):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_args = {
        'lr': 0.001,
        'batch_size': 32,
        'max_epochs': 100,
        'n_gpu': 1,
        'hidden_size': 128,
        'num_layers': 16
    }
    
    vad_args = Namespace(**vad_args)
    vad_path = os.path.join(APP_ROOT, "checkpoints/vad_model.ckpt")
    vad_model = VADModel.load_from_checkpoint(vad_path, args=vad_args)
    vad_model.to(device)
    
    
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
    audio_tensor = audio_tensor.to(device)
    mel = vad_model.spec_converter(audio_tensor)
    
    
    out = vad_model(mel)
    out = vad_model.sigmoid(out)

    out = (out > threshold).float().cpu()

    out = get_speech_interval(out, min_dur)
    
    ddk_rate, ddk_avg, ddk_std, ddk_pause_rate, pause_avg, ddk_pause_std = cal_features_from_speech_interval(out, dur)
    
    return ddk_rate, ddk_avg, ddk_std, ddk_pause_rate, pause_avg, ddk_pause_std

def get_phonation_features(path):
    ddk_path = os.path.join(APP_ROOT, "ddk")
    ddk_features = DDK(path, ddk_path)
    ddk_features = [0 if isnan(x) else x for x in ddk_features]
    
    return ddk_features

def get_intelligibility(path):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
    
    intel_args = {
        'lr': 0.001,
        'batch_size': 32,
        'max_epochs': 100,
        'n_gpu': 1,
    }
    
    intel_args = Namespace(**intel_args)
    
    intel_path = os.path.join(APP_ROOT, "checkpoints/intelligibility_model.ckpt")
    #self.model = PTKModel(n_classes=5, args=args)
    intel_model = PTKModel.load_from_checkpoint(intel_path,n_classes=5,args=intel_args)
    intel_model.to(device)
    intel_model.eval()
    
    y = prepare_data(path)

    y = my_padding(y, 16000 * 15).to(device)
    y = intel_model.mel_converter(y)
    y = y.unsqueeze(0)
    y = y.unsqueeze(0)
    y = intel_model(y)
    
    intelligibility = torch.argmax(y, dim=-1).cpu().numpy()
    
    return int(intelligibility[0])
    
def get_features(path, gender):
    intelligibility = get_intelligibility(path)
    phonation_features = get_phonation_features(path)
    
    threshold = 0.7
    min_dur = 0.08
    prosody_respiration_features = get_prosody_respiration_features(path, threshold, min_dur)
    
    features = [gender, intelligibility]
    features += phonation_features
    features += prosody_respiration_features
    
    # features = [intelligibility]
    # features += phonation_features
    # features += [gender]
    # features += prosody_respiration_features
    
    return features
    
    
if __name__ == "__main__":
    path = "/mnt/code/xai_wav2vec_exp/nia_HC0002_50_0_1_5_002_1.wav"
    gender = 0
    
    features = get_features(path, gender)
    print(features)
    