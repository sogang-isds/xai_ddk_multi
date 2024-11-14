from multi_input_model import DDKWav2VecModel
from get_features import get_features, my_padding

import torchaudio
import torch

from argparse import ArgumentParser
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(wav):
    """
    First time setup of the dataset
    """

    y, sr = torchaudio.load(wav)  # type:ignore
    
    if sr != 16000:
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=16000)
    
    y = y.mean(0)
    
    y = my_padding(y, 16000*15)

    return y.unsqueeze(0)

def get_severity(model, audio_filepath, gender):    
    
    audio = prepare_data(audio_filepath).to(device)
    mel_x = model.mel_spectrogram(audio.unsqueeze(0))
    mel_x = model.db_converter(mel_x)
    
    feature = get_features(audio_filepath, gender)
    scaler = joblib.load("scaler.save")
    feature = scaler.transform([feature])
    feature = torch.tensor(feature).float().to(device)
    
    out = model(mel_x, audio, feature)
    
    severity = torch.argmax(out, dim=-1)[0]
    
    return severity



if __name__ == "__main__":
    
    args = ArgumentParser()
    args.add_argument("--audio_filepath", type=str, default="./nia_HC0002_50_0_1_5_002_1.wav", help="절대경로로 입력")
    args.add_argument("--gender", type=str, default="F", help="F or M")
    
    args = args.parse_args()
    
    idx2sev = { 0 : 'normal',
               1 : 'mild-to-moderate',
               2 : 'severe'
        
    }
    
    audio_filepath = args.audio_filepath
    gender = 0 if args.gender == 'M' else 1
    
    model = DDKWav2VecModel.load_from_checkpoint("./checkpoints/multi_input_model.ckpt").to(device).eval()
    
    
    severity = get_severity(model, audio_filepath, gender)
    print(f"[Inference Result] {severity} ({idx2sev[int(severity)]})")
    
