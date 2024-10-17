import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay
from torchaudio.transforms import Spectrogram, AmplitudeToDB
from math import ceil
import random
import matplotlib.pyplot as plt
import pdb
import numpy as np

class VADModel(pl.LightningModule):
    def __init__(self, args=None):
        
        super().__init__()
        # self.save_hyperparameters(args)
        # self.lr = args.lr
        self.batch_size = args.batch_size
        self.n_gpu = args.n_gpu
        self.sample_rate = 16000 
        self.win_length = int(self.sample_rate * 0.01)  # 10ms
        self.hop_length = int(self.sample_rate * 0.0025)  # 2.5ms
        self.spec_converter = Spectrogram( 
                                        n_fft=256, 
                                        win_length=self.win_length, 
                                        hop_length=self.hop_length)
        self.db_converter = AmplitudeToDB()

        # self.dataset = VADDataset(args.datadir)
        # self.trainset = VADDataset(args.datadir, "/xai/ddk_task/ddk_train.csv")
        # self.valset = VADDataset(args.datadir, "/xai/ddk_task/ddk_val.csv")
        # self.testset = VADDataset(args.datadir, "/xai/ddk_task/ddk_test.csv")
        # self.train_idx, self.val_idx, self.test_idx = self.train_val_split()

        self.snr_levels=[15,20,25,100]
        self.num_masks=20
        self.mask_lengths = [200,300,400,500]
        
        self.validation_step_outputs = []  #for on_validation_end
        self.state = []
        self.test_step_output = []

        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.lstm1 = nn.LSTM(129, self.hidden_size, self.num_layers, batch_first=True)
        # self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size,1)
            # nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy(threshold=0.5)
        
    def time_to_mel_frame(self, time):
        return int(time * self.sample_rate / self.hop_length)
        
    def add_noise(self, audio, snr_levels, num_masks, mask_lengths):
        """
        Add noise to the audio with the given Signal-to-Noise Ratio (SNR).
        
        Parameters:
            audio (torch.Tensor): The clean audio signal
            noise (torch.Tensor): The noise signal
            snr (float): The desired SNR in dB
            s
        Returns:
            torch.Tensor: The noisy audio signal
        """
        # Calculate the energy of the signals
        noise = torch.rand_like(audio).to("cuda")
        snr = random.choice(snr_levels)
        
        audio_energy = torch.norm(audio, dim=-1)
        noise_energy = torch.norm(noise, dim=-1)

        # Calculate the scaling factor for noise
        scale_factor = (audio_energy / noise_energy) * (10 ** (-snr / 20))

        # Scale the noise
        scaled_noise = scale_factor.view(-1, 1) * noise
        if len(scaled_noise) > audio.shape[-1]:
            scaled_noise = scaled_noise[:audio.shape[-1]]

        # Apply random time masking on the scaled noise
        for _ in range(num_masks):
            mask_length = random.choice(mask_lengths)

            mask_start = random.randint(0, scaled_noise.shape[-1] - mask_length)
            scaled_noise[mask_start:mask_start+mask_length] = 0
            
        # Add the scaled and masked noise to the original audio
        noisy_audio = audio + scaled_noise
        
        return noisy_audio

        
    def train_val_split(self):
        #split train, valid and test set ratio 8:1:1
        train_idx,tmp_idx,train_y,tmp_y = train_test_split(torch.arange(len(self.dataset)),self.dataset.labels,
                                             test_size=0.2,
                                             random_state=999,
                                             shuffle=True)
        valid_idx,test_idx,valid_y,test_y = train_test_split(tmp_idx,tmp_y,
                                             test_size=0.5,
                                             random_state=999,
                                             shuffle=True)
        return train_idx,valid_idx,test_idx
    
    def train_dataloader(self):
        # train_dataset = Subset(self.dataset, self.train_idx)
        return DataLoader(self.trainset, batch_size=self.batch_size,
                                           num_workers=4, shuffle=True)
    def val_dataloader(self):
        # valid_dataset = Subset(self.dataset, self.val_idx)
        return DataLoader(self.valset, batch_size=self.batch_size,
                                           num_workers=4, shuffle=False)
        
    def test_dataloader(self):
        # test_dataset = Subset(self.dataset, self.test_idx)
        return DataLoader(self.testset, batch_size=self.batch_size,
                                           num_workers=4, shuffle=False)
        
    def forward(self, x):
        x = self.db_converter(x)
        x = x.transpose(1,2).contiguous()
        
        out, _ = self.lstm1(x)
        # out, _ = self.lstm2(out)
        # out, _ = self.lstm3(out)
        out = self.tanh(out)
        out = self.fc(out)
        out = out.squeeze()
        # out = (out > 0.5).float()
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.add_noise(x, self.snr_levels, self.num_masks, self.mask_lengths)
        x = self.spec_converter(x.detach())
        
        x = self(x)
        
        loss = self.loss(x, y)
        self.log("train/loss", loss, prog_bar=True)
        ret = {'loss': loss, 'train/pred' : x, 'train/true' : y}
        self.state.append(ret)
        
        return ret
    
    def on_train_epoch_end(self) -> None:
        outputs = self.state            
        acc = self.accuracy(self.sigmoid(torch.cat([x['train/pred'] for x in outputs])),torch.cat([x['train/true'] for x in outputs]))

        self.log("train/acc", acc,prog_bar=True,on_epoch=True,sync_dist=True)
        self.state.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        x = self.spec_converter(x.detach())
        x = self(x)
        
        loss = self.loss(x, y)
        acc = self.accuracy(self.sigmoid(x), y)
        ret = {'val_loss': loss, 'val_acc' : acc, 'preds' : x, 'true' : y}
        self.validation_step_outputs.append(ret)
        
        return ret
    
    def on_validation_epoch_end(self):
        
        # OPTIONAL        
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = self.accuracy(self.sigmoid(torch.cat([x['preds'] for x in outputs])),torch.cat([x['true'] for x in outputs]))

        self.log("val/loss", avg_loss,prog_bar=True,on_epoch=True)
        self.log("val/acc", acc,prog_bar=True,on_epoch=True,sync_dist=True)
            
        self.validation_step_outputs.clear()
        return {'val/loss': avg_loss, 'val/acc' : acc}
    
    def test_step(self, batch, batch_idx ) :
        x, y = batch
        x = self.spec_converter(x)
        x = self(x)           
        loss = self.loss(x, y)
        acc = self.accuracy(self.sigmoid(x), y)
        ret = {'test_loss': loss, 'test_acc' : acc, 'test_preds' : x, 'test_true' : y}
        self.test_step_output.append(ret)        
        
        return ret
    
    def on_test_epoch_end(self):
        outputs = self.test_step_output

        acc = self.accuracy(self.sigmoid(torch.cat([x['test_preds'] for x in outputs])),torch.cat([x['test_true'] for x in outputs]))
        
        print(acc)
        plt.figure(figsize=(8,8))
        y_true = torch.cat([x['test_true'] for x in outputs])
        y_pred = self.sigmoid(torch.cat([x['test_preds'] for x in outputs]))
        y_true = y_true.reshape(1,-1)
        y_pred = y_pred.reshape(1,-1)
        print(y_true.shape)
        print(y_true[0])
        print(self.accuracy(y_pred[0], y_true[0]))
        # RocCurveDisplay.from_predictions(y_true[0].cpu().numpy(), y_pred[0].cpu().numpy())
        fpr, tpr, thresholds = roc_curve(y_true[0], y_pred[0])
        score = roc_auc_score(y_true[0], y_pred[0])
        plt.plot(fpr, tpr, linewidth=3, label=f"ROC(AUC:{score:.2f})")
        # fig.setp(linewidth=3.0)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('FPR')
        plt.ylabel('TPR( Recall )')
        plt.legend()
        # plt.savefig("vad_testset_roc.png")
        plt.savefig("vad_testset_roc.png")
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("Threshold value is:", optimal_threshold)
         

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(
            limit_batches * batches)

        num_devices = max(1,self.n_gpu,self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        print("==================================")
        print(batches,effective_accum,self.trainer.max_epochs)
        print("==================================")
        return (batches // effective_accum) * self.trainer.max_epochs # type: ignore

    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        #warmup and decay lr scheduler
        # print("==================================")
        # print(len(self.train_dataloader()),self.batch_size,self.trainer.accumulate_grad_batches)
        # print("==================================")
        steps_per_epoch = ceil(len(self.train_dataloader())/(self.trainer.accumulate_grad_batches))
        #self.trainer.num
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.lr,pct_start=0.05, steps_per_epoch=steps_per_epoch, epochs=self.trainer.max_epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.lr,pct_start=0.03,
                                                        epochs=self.trainer.max_epochs,steps_per_epoch=steps_per_epoch
                                                        ,anneal_strategy='linear')
        return [optim], [scheduler]

def plot_item(data):
    audio_tensor, mel_tensor, label_tensor = data
    print("shape")
    print("audio_tensor, mel_tensor, label_tensor")
    print(audio_tensor.shape, mel_tensor.shape, label_tensor.shape)
    plt.figure(figsize=(15, 6))

    # Plot waveform
    plt.subplot(2, 1, 1) 
    plt.plot(audio_tensor.numpy())
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Create a twin Axes sharing the xaxis
    ax2 = plt.twinx().twiny()

    # Plot label tensor on the new Axes
    ax2.plot(label_tensor.numpy(), 'r')
    ax2.set_ylabel('Voice/Unvoice Label', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    # Plot Mel Spectrogram
    plt.subplot(2, 1, 2)
    plt.imshow(mel_tensor.numpy(), aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig("example.png")

if __name__=="__main__":
    args = ArgumentParser()
    
    args.add_argument("--prj",type=str,default="resnext")
    args.add_argument("--lr",type=float,default=0.001)
    args.add_argument("--batch_size",type=int,default=4)
    args.add_argument('--max_epochs',type=int,default=100)
    args.add_argument('--n_gpu',type=int,default=1)
    args.add_argument('--datadir', type=str, default="/mnt/nas/CORPUS/XAI_dataset/relabel/v2_말명료도_재레이블링/002_DDK/only_ddk")
    args.add_argument('--hidden_size', type=int, default=40)
    args.add_argument('--num_layers', type=int, default=4)
    
    args = args.parse_args()
    
    model = VADModel(args)
    train = model.train_dataloader()
    audio, label= next(iter(train))
    
    mel = model.mel_converter(audio.detach())
    plot_item((audio[1], mel[1], label[1]))
    
    # out = model(audio)
    # pdb.set_trace()
    # print(out)
    # print(out.shape)
