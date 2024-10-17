import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import models
import torchaudio.transforms as T
#from resnet1d import ResNet1D
from sklearn.model_selection import train_test_split
import wandb
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from torchaudio.transforms import TimeMasking,FrequencyMasking,MelSpectrogram
from math import ceil
# pytorch transformer model for classification

    
        


# write code lightning code for train
class PTKModel(pl.LightningModule):
    def __init__(self,n_classes=5,args=None):
        
        super().__init__()
        
        self.n_classes = n_classes
        self.lr = args.lr #if args != None else args.lr
        self.batch_size = args.batch_size #if args != None else args.batch_size
        self.logging = True #if args != None else args.logging
        self.img_logging = False #if args != None else args.img_logging
        self.n_gpu = args.n_gpu#if args != None else args.n_gpu        
        _window_size = int(16000 * 0.025)
        _hop_size = int(16000 * 0.01)     
         
        self.speed_pertubation = T.SpeedPerturbation(16000,factors=[0.95,1.05,1.0,1.0])
        self.spec_aug = nn.Sequential(FrequencyMasking(freq_mask_param=22),
                                        TimeMasking(time_mask_param=30),
                                        FrequencyMasking(freq_mask_param=22),
                                        TimeMasking(time_mask_param=30))
        self.mel_converter = MelSpectrogram(sample_rate=16000, 
                                            n_fft=1024,
                                            win_length=_window_size, 
                                            hop_length=_hop_size, 
                                            n_mels=224,
                                            normalized=True
                                            )

        # self.trainset = PTKDataset("./data",setup=False)
        self.macro_avg_metric = MulticlassAccuracy(num_classes=n_classes,average='macro')
        self.micro_avg_metric = MulticlassAccuracy(num_classes=n_classes,average='micro')        
        self.validation_step_outputs = []  #for on_validation_end
        self.state = []
        self.test_step_output = []

        # self.labels = self.trainset.labels
        # self.class_weights = torch.tensor(compute_class_weight(class_weight='balanced',
        #                                                        classes=np.unique(self.labels),
        #                                                        y=self.labels),dtype=torch.float32)
        self.class_weights = torch.tensor([0,0,0,0,0], dtype = torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights,reduction='mean')
        #convolutions layer for feature extraction
        self.model = models.resnext101_32x8d(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.model.fc = nn.Linear(2048,n_classes)       #resnet is 512, resnext is 2048
        # self.train_idx, self.val_idx, self.test_idx = self.train_val_split()
        
    
       
    
        
    def cm_view(self,true_y,pred_y,subset="Validation",epoch=0,savename="confusion_matrix.png"):
        cf = confusion_matrix(true_y,pred_y)
        acc = accuracy_score(true_y,pred_y)
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
        plt.title(f"macro_avg_acc: {macro_avg_acc*100:.2f}%,  micro_avg_acc: {micro_avg_acc*100:.2f}%",fontsize=8)
        plt.ylabel("True Label")
        plt.xlabel("Predict Label")           
        plt.savefig(savename)
        plt.tight_layout()
        plt.close()
    

    def train_val_split(self):
        #split train, valid and test set ratio 8:1:1
        train_idx,tmp_idx,train_y,tmp_y = train_test_split(torch.arange(len(self.trainset)),self.trainset.labels,
                                             test_size=0.2,
                                             random_state=999,
                                             shuffle=True,
                                             stratify=self.trainset.labels)
        valid_idx,test_idx,valid_y,test_y = train_test_split(tmp_idx,tmp_y,
                                             test_size=0.5,
                                             random_state=999,
                                             shuffle=True,
                                             stratify=tmp_y)
        return train_idx,valid_idx,test_idx
        
    
    def forward(self,x):
        
        x = self.model(x)
        #print(x.shape)        
        return x
    def spec_show(self,x,y):
        fig = plt.figure()
        spec = x[0].detach().cpu().numpy().squeeze(0)        
        ax = plt.gca()
        ax.imshow(spec)
        ax.set_title(f"true: {y[0]}")
        fig.savefig("spec.png")
        
        
    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        # x = self.speed_pertubation(x.detach())[0]
        x = self.mel_converter(x.detach())
        x = self.spec_aug(x.detach())
        
        if self.img_logging :
            self.spec_show(x,y)
            self.logger.experiment.log({"train_spec":wandb.Image("spec.png")}) #require plotly
        
        x = self(x)
        #x = nn.functional.softmax(x,dim=-1)
        
        loss = self.loss(x, y)
        self.log("train/loss", loss,prog_bar=True)
        ret = {'loss': loss, 'train/pred' : torch.argmax(x,dim=-1), 'train/true' : y}
        self.state.append(ret)
        
        return ret
    
        
    def on_train_epoch_end(self) -> None:
        outputs = self.state            
        macro_avg_acc = self.macro_avg_metric(torch.cat([x['train/pred'] for x in outputs]),torch.cat([x['train/true'] for x in outputs]))
        micro_avg_acc = self.micro_avg_metric(torch.cat([x['train/pred'] for x in outputs]),torch.cat([x['train/true'] for x in outputs]))
        #self.log("val/loss", avg_loss,prog_bar=True,on_epoch=True)
        self.log("train/macro_acc", macro_avg_acc,prog_bar=True,on_epoch=True,sync_dist=True)
        self.log("train/micro_acc", micro_avg_acc,prog_bar=True,on_epoch=True,sync_dist=True)  
        self.state.clear()
        
        

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        
        x = self.mel_converter(x.detach())        
        x = self(x)
        #x = nn.functional.softmax(x,dim=-1)        
        loss = self.loss(x, y)
        acc = torch.sum(torch.argmax(x,dim=-1)==y).item()/len(y)
        ret = {'val_loss': loss, 'val_acc' : acc, 'preds' : torch.argmax(x,dim=-1), 'true' : y}
        self.validation_step_outputs.append(ret)
        
        
        return ret 
    
    def on_validation_epoch_end(self):
        
        # OPTIONAL        
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        macro_avg_acc = self.macro_avg_metric(torch.cat([x['preds'] for x in outputs]),torch.cat([x['true'] for x in outputs]))
        micro_avg_acc = self.micro_avg_metric(torch.cat([x['preds'] for x in outputs]),torch.cat([x['true'] for x in outputs]))
        #val_acc = torch.cat([x['val_acc'] for x in outputs]).mean()
        pred = torch.cat([x['preds'] for x in outputs])
        true = torch.cat([x['true'] for x in outputs])
        self.log("val/loss", avg_loss,prog_bar=True,on_epoch=True)
        self.log("val/macro_acc", macro_avg_acc,prog_bar=True,on_epoch=True,sync_dist=True)
        self.log("val/micro_acc", micro_avg_acc,prog_bar=True,on_epoch=True,sync_dist=True)      
        #logging confusion matrix using pl logger
        if self.logging and self.n_gpu == 1:
            self.cm_view(true.cpu().numpy(),pred.cpu().numpy(),subset="Validation")
            self.logger.experiment.log({"cm":wandb.Image("confusion_matrix.png")}) #require plotly
        #self.logger.experiment.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None, y_true=true.cpu().numpy(), preds=pred.cpu().numpy(), class_names=[str(x) for x in range(self.n_classes)])})
        self.validation_step_outputs.clear()
        return {'val/loss': avg_loss, 'val/macro_acc' : macro_avg_acc, 'val/micro_acc' : micro_avg_acc}
        
    def test_step(self, batch,batch_idx ) :
        x, y = batch
        x = self.mel_converter(x)
        x = self(x)           
        loss = self.loss(x, y)
        acc = torch.sum(torch.argmax(x,dim=-1)==y).item()/len(y)
        ret = {'test_loss': loss, 'test_acc' : acc, 'test_preds' : torch.argmax(x,dim=-1), 'test_true' : y}
        self.test_step_output.append(ret)        
        
        return ret
    
    def on_test_epoch_end(self):
        outputs = self.test_step_output
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        macro_avg_acc = self.macro_avg_metric(torch.cat([x['test_preds'] for x in outputs]),torch.cat([x['test_true'] for x in outputs]))
        micro_avg_acc = self.micro_avg_metric(torch.cat([x['test_preds'] for x in outputs]),torch.cat([x['test_true'] for x in outputs]))
        #val_acc = torch.cat([x['val_acc'] for x in outputs]).mean()
        pred = torch.cat([x['test_preds'] for x in outputs])
        true = torch.cat([x['test_true'] for x in outputs])
        #logging confusion matrix using pl logger
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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.lr,pct_start=0.05,
                                                        epochs=self.trainer.max_epochs,steps_per_epoch=steps_per_epoch
                                                        ,anneal_strategy='linear')
        return [optim], [scheduler]

    
    def train_dataloader(self):
        # REQUIRED
        train_dataset = Subset(self.trainset, self.train_idx)
        # train_dataset.dataset.pre_transform = self.speed_pertubation
        # train_dataset.dataset.transform = self.spec_aug
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                           num_workers=4,shuffle=True
                                           )
    def val_dataloader(self) :
        valid_dataset = Subset(self.trainset, self.val_idx)
        return torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size,
                                           num_workers=4,shuffle=False,
                                           )
    
    def test_dataloader(self) :
        test_dataset = Subset(self.trainset, self.test_idx)        
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                           num_workers=4,shuffle=False,)
    

