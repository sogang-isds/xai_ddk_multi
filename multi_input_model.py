import os
from common import APP_ROOT
from multi_input_resnet_model import DDK_ResNet

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from torchmetrics.classification import  MulticlassAccuracy
import torchaudio.transforms as T
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision import models
import torch.nn.functional as F

from transformers import Wav2Vec2Model

from argparse import ArgumentParser
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
import pdb
import joblib


class DDKWav2VecModel(pl.LightningModule):
    def __init__(self,args=None):
        
        super().__init__()
        self.save_hyperparameters(args)
        
        self.n_classes = self.hparams.n_classes
        self.lr = self.hparams.lr
        self.batch_size = self.hparams.batch_size
        self.n_gpu = self.hparams.n_gpu
        self.sample_rate = 16000 
        self.win_length = int(self.sample_rate * 0.025)  # 25ms
        self.hop_length = int(self.sample_rate * 0.02)  # 20ms
        self.mel_spectrogram = MelSpectrogram(sample_rate=self.sample_rate, 
                                              n_fft=512, 
                                              win_length=self.win_length, 
                                              hop_length=self.hop_length, 
                                              n_mels=80)
        self.db_converter = AmplitudeToDB()

        self.labels = np.load(os.path.join(APP_ROOT, "labels.npy"))
        self.labels = torch.from_numpy(self.labels)
        
        # self.validation_step_outputs = []  #for on_validation_end
        # self.state = []
        # self.test_step_output = []

        # CNN for spectrogram
        self.resnet_model = DDK_ResNet(num_classes=3)
        self.post_spec_layer = nn.Linear(2048, 749)
        self.post_spec_batchnorm = nn.BatchNorm1d(749)
        self.post_spec_dropout = nn.Dropout(p=0.3)
        
        # Wav2Vec 2.0
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.wav2vec.freeze_feature_encoder()
        self.post_w2v_layer = nn.Linear(self.wav2vec.config.hidden_size, 768)
        self.post_w2v_batchnorm = nn.BatchNorm1d(768)
        self.post_wv2_droput = nn.Dropout(p=0.3)
        
        # CNN / Wav2Vec 2.0 combination
        self.post_attn_layer = nn.Linear(1517, 128)
        self.post_attn_batchnorm = nn.BatchNorm1d(128)
        self.post_attn_dropout = nn.Dropout(p=0.3)
        
        
        # DNN for characteristics
        self.char_layer = nn.Sequential(
            nn.Linear(13, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        
        #combination
        self.final_layer = nn.Sequential(
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),    
            
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3)
        )
        
        self.relu = nn.ReLU()
        
        self.class_weights = torch.tensor(compute_class_weight(class_weight='balanced',
                                                               classes=np.unique(self.labels),
                                                               y=self.labels.numpy()),dtype=torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights,reduction='mean')
        
        # self.macro_avg_metric = MulticlassAccuracy(num_classes=self.n_classes,average='macro')
        # self.micro_avg_metric = MulticlassAccuracy(num_classes=self.n_classes,average='micro')
        
    def cm_view(self,true_y,pred_y,subset="Validation",epoch=0,savename="confusion_matrix.png"):
        cf = confusion_matrix(true_y,pred_y)
        macro_avg_acc = self.macro_avg_metric(torch.tensor(pred_y).cuda(),torch.tensor(true_y).cuda())
        micro_avg_acc = self.micro_avg_metric(torch.tensor(pred_y).cuda(),torch.tensor(true_y).cuda())
        
        fig, ax = plt.subplots()
        annot = np.empty_like(cf).astype(str)
        cf_sum = np.sum(cf, axis=1, keepdims=True)
        cf_perc = cf/cf.astype(np.float64).sum(axis=1)[:,None]*100

        nrows, ncols = cf.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cf[i, j]
                p = cf_perc[i, j]
                if i == j:
                    s = cf_sum[i]
                    annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
                #elif c == 0:
                #    annot[i, j] = ''
                else:
                    annot[i, j] = '%.2f%%\n%d' % (p, c)


        label = [str(x) for x in range(self.n_classes)]
        sns.heatmap(cf_perc,annot=annot, cmap='Blues',fmt='',cbar_kws={'format':PercentFormatter()},xticklabels=label,yticklabels=label,vmin=0,vmax=100,ax=ax)
        title_text = f"{subset} Confusion Matrix " 
        if epoch > 0 :
            title_text = title_text + f"on epoch: {epoch}"
        plt.suptitle(title_text,y=1.0)
        plt.title(f"balanced_accuracy: {macro_avg_acc*100:.2f}%,  accuracy: {micro_avg_acc*100:.2f}%",fontsize=8)
        # plt.title(f"accuracy: {micro_avg_acc*100:.2f}%",fontsize=8)
        plt.ylabel("True Label")
        plt.xlabel("Predict Label")           
        plt.savefig(savename)
        plt.tight_layout()
        plt.close()

    
    # def train_dataloader(self):
    #     return DataLoader(self.trainset, batch_size=self.batch_size,
    #                                        num_workers=0, shuffle=True, collate_fn=PadSequence())
    # def val_dataloader(self):
    #     return DataLoader(self.valset, batch_size=self.batch_size,
    #                                        num_workers=0, shuffle=False, collate_fn=PadSequence())
        
    # def test_dataloader(self):
    #     return DataLoader(self.testset, batch_size=self.batch_size,
    #                                        num_workers=0, shuffle=False, collate_fn=PadSequence())
        
    def forward(self, spec_x, w2v_x, char_x):

        
        # CNN(ResNet)
        spec_x, _ = self.resnet_model(spec_x)
        
        # CNN projection
        spec_x = self.post_spec_layer(spec_x)
        spec_x = self.relu(spec_x)
        spec_x = self.post_spec_dropout(spec_x)

        spec_attn_x = spec_x.reshape(spec_x.shape[0], 1, -1)
        
        # wav2vec 2.0 
        w2v_x = self.wav2vec(w2v_x)[0]
        w2v_x = torch.matmul(spec_attn_x, w2v_x)
        w2v_x = w2v_x.reshape(w2v_x.shape[0], -1)
        
        # wav2vec projection
        w2v_x = self.post_w2v_layer(w2v_x)
        w2v_x = self.relu(w2v_x)
        w2v_x = self.post_wv2_droput(w2v_x)
        
        # CNN + wav2vec concat and projection
        spec_w2v_x = torch.cat([spec_x, w2v_x], dim=-1)
        spec_w2v_x = self.post_attn_layer(spec_w2v_x)
        spec_w2v_x = self.relu(spec_w2v_x)
        spec_w2v_x = self.post_attn_dropout(spec_w2v_x)
        
        
        # Characteristics projection
        char_x = self.char_layer(char_x)
        
        # CNN + wav2vec + characteristics concat
        total_x = torch.cat([spec_w2v_x, char_x], dim=-1)
        
        # Final output
        out = self.final_layer(total_x)
        
        return out
    
    def _shared_step(self, batch):
        audio, feature, y = batch
        
        mel_x = self.mel_spectrogram(audio)
        mel_x = self.db_converter(mel_x)
        audio = audio.squeeze(1)
        
        out = self(mel_x, audio, feature)
        loss = self.loss(out, y)
        
        return loss, y, out
    
    def training_step(self, batch, batch_idx):
        loss, y, out = self._shared_step(batch)
        
        self.log("train/loss", loss, prog_bar=True)
        ret = {'loss': loss, 'train/pred' : torch.argmax(out, dim=-1), 'train/true' : y}
        self.state.append(ret)
        
        return ret
    
    def on_train_epoch_end(self) -> None:
        outputs = self.state      
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()      
        macro_avg_acc = self.macro_avg_metric(torch.cat([x['train/pred'] for x in outputs]),torch.cat([x['train/true'] for x in outputs]))
        micro_avg_acc = self.micro_avg_metric(torch.cat([x['train/pred'] for x in outputs]),torch.cat([x['train/true'] for x in outputs]))

        self.log("train/loss", avg_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train/macro_acc", macro_avg_acc,prog_bar=True,on_epoch=True,sync_dist=True)
        self.log("train/micro_acc", micro_avg_acc,prog_bar=True,on_epoch=True,sync_dist=True) 
        
        self.state.clear()
    
    def validation_step(self, batch, batch_idx):
        loss, y, out = self._shared_step(batch)
        
        ret = {'val_loss': loss, 'preds' : torch.argmax(out, dim=-1), 'true' : y}
        self.validation_step_outputs.append(ret)
        
        return ret
    
    def on_validation_epoch_end(self):
        
        # OPTIONAL        
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        macro_avg_acc = self.macro_avg_metric(torch.cat([x['preds'] for x in outputs]),torch.cat([x['true'] for x in outputs]))
        micro_avg_acc = self.micro_avg_metric(torch.cat([x['preds'] for x in outputs]),torch.cat([x['true'] for x in outputs]))
        
        pred = torch.cat([x['preds'] for x in outputs])
        true = torch.cat([x['true'] for x in outputs])

        self.log("val/loss", avg_loss,prog_bar=True,on_epoch=True)
        self.log("val/macro_acc", macro_avg_acc,prog_bar=True,on_epoch=True,sync_dist=True)
        self.log("val/micro_acc", micro_avg_acc,prog_bar=True,on_epoch=True,sync_dist=True)    
        
        if self.n_gpu == 1:
            self.cm_view(true.cpu().numpy(),pred.cpu().numpy(),subset="Validation")
            self.logger.experiment.log({"cm":wandb.Image("confusion_matrix.png")})
            
        self.validation_step_outputs.clear()
        return {'val/loss': avg_loss, 'val/macro_acc' : macro_avg_acc, 'val/micro_acc' : micro_avg_acc}
    
    def test_step(self, batch, batch_idx ) :
        loss, y, out = self._shared_step(batch)

        ret = {'test_loss': loss, 'test_preds' : torch.argmax(out, dim=-1), 'test_true' : y}
        self.test_step_output.append(ret)        
        
        return ret
    
    def on_test_epoch_end(self):
        outputs = self.test_step_output

        macro_avg_acc = self.macro_avg_metric(torch.cat([x['test_preds'] for x in outputs]),torch.cat([x['test_true'] for x in outputs]))
        micro_avg_acc = self.micro_avg_metric(torch.cat([x['test_preds'] for x in outputs]),torch.cat([x['test_true'] for x in outputs]))
        
        print(f"micro average accuracy : {micro_avg_acc}")
        print(f"macro average accuracy : {macro_avg_acc}")
        
        pred = torch.cat([x['test_preds'] for x in outputs])
        true = torch.cat([x['test_true'] for x in outputs])
        
        self.cm_view(true.cpu().numpy(),pred.cpu().numpy(),subset="Test set",epoch=-1,savename="test_confusion_matrix.png")  

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
        # print("==================================")
        # print(batches,effective_accum,self.trainer.max_epochs)
        # print("==================================")
        return (batches // effective_accum) * self.trainer.max_epochs # type: ignore

    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999),  # 모멘트 추정치의 계수
            eps=1e-8,  # 수치적 안정성을 위한 작은 값
            weight_decay=0.01,  # 가중치 감쇠 계수
            amsgrad=False)
        #warmup and decay lr scheduler
        print("==================================")
        print(len(self.train_dataloader()),self.batch_size,self.trainer.accumulate_grad_batches)
        print("==================================")
        steps_per_epoch = ceil(len(self.train_dataloader())/(self.trainer.accumulate_grad_batches))
        #self.trainer.num
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.lr,pct_start=0.05, steps_per_epoch=steps_per_epoch, epochs=self.trainer.max_epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.lr,pct_start=0.03,
                                                        epochs=self.trainer.max_epochs,steps_per_epoch=steps_per_epoch
                                                        ,anneal_strategy='linear')
        return [optim], [scheduler]
    
    
if __name__=="__main__":
    args = ArgumentParser()
    #add project name to args
    args.add_argument("--prj",type=str,default="resnext_ddk_severity")
    args.add_argument("--lr",type=float,default=0.001)
    args.add_argument("--batch_size",type=int,default=16)
    args.add_argument('--max_epochs',type=int,default=100)
    args.add_argument('--n_gpu',type=int,default=1)
    args.add_argument('--n_classes', type=int, default=3)
    args = args.parse_args()
    
    # testset = DDKDataset("/xai/ddk_task/ddk_test.csv")
    # testloader = DataLoader(testset, batch_size=args.batch_size,
    #                                        num_workers=4, shuffle=False)
    
    # audio, label = next(iter(testloader))
    
    model = DDKWav2VecModel( args = args).cuda()
    testloader = model.test_dataloader()
    audio, feature, label = next(iter(testloader))
    audio = audio.cuda()
    feature = feature.cuda()
    
    mel_x = model.mel_spectrogram(audio) 
    audio = audio.squeeze(1)
    
    # print(audio.shape)
    # print(mel_x.shape)
    # print(feature.shape)
    out = model(mel_x, audio)
    # print(out)
    # print(out.shape)
    
    pdb.set_trace()