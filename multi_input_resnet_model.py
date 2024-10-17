import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchaudio.transforms import MelSpectrogram
import pytorch_lightning as pl

class DDK_ResNet(pl.LightningModule):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.n_classes = num_classes
        
        self.sample_rate = 16000 
        self.win_length = int(self.sample_rate * 0.025)  # 25ms
        self.hop_length = int(self.sample_rate * 0.02)  # 20ms
        self.mel_spectrogram = MelSpectrogram(sample_rate=self.sample_rate, 
                                              n_fft=512, 
                                              win_length=self.win_length, 
                                              hop_length=self.hop_length, 
                                              n_mels=80)
        
        self.weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet.fc = nn.Linear(2048, self.n_classes)
        
        
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.fc = self.resnet.fc
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.save_hyperparameters("num_classes")
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        
        return x, out
    
    
    