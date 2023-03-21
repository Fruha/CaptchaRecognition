import os
import pytorch_lightning as pl
import json
import cv2
import torch
import textdistance
from matplotlib import pyplot as plt

from torch import optim, nn, utils, Tensor
from torchvision import transforms as T
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import Dataset

from torchvision import models
from utils import load_from_json, decode_str, ctc_decoder


class CaptchaDataset(Dataset):
    def __init__(self, dir_path: str, type_: str, char_set_pred: str, hparams:dict, transform=None):
        self.hparams = hparams
        self.type = type_
        self.dir_path = dir_path
        self.labels = load_from_json(os.path.join(self.dir_path, 'labels.json'))
        self.labels = {int(key):val for key, val in self.labels.items()}
        self.images = list(filter(lambda x: x.find('.png') != -1, os.listdir(dir_path)))
        self.transform = transform
        self.char_set_pred = char_set_pred
        self.encoder = {char: num for num, char in enumerate(char_set_pred)}
        self.decoder = {num: char for num, char in enumerate(char_set_pred)}

    def __len__(self):
        if self.type == 'train':
            return len(self.images[:self.hparams['max_count_dataset_train']])
        elif self.type == 'val':
            return len(self.images[:self.hparams['max_count_dataset_val']])
        

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, f'{idx}.png')
        image = cv2.imread(img_path)
        if self.transform:
            image = self.transform(image=image)

        text_length = len(self.labels[idx])
        text = self.labels[idx].ljust(self.hparams['max_length_output'])
        tokens = torch.LongTensor([self.encoder[char] for char in text])

        return image['image'], tokens, text_length




class CaptchaModel(pl.LightningModule):
    def __init__(self, hparams:dict, char_set_pred:str):
        super(CaptchaModel, self).__init__()
        self.char_set_pred = char_set_pred
        self.hparams.update(hparams)

        self.encoder = nn.Sequential(*[models.efficientnet_b0().features[i] for i in range(3)])
        self.fc_1 = nn.Sequential(
            nn.Linear(552,128), #352
            nn.Dropout(0.2),
            nn.BatchNorm1d(70),
        )
        # self.lstm = nn.LSTM(128, 64, num_layers=1, batch_first=True, bidirectional=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.fc_2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(128,len(self.char_set_pred)),
        )

        self.loss_fn = getattr(torch.nn, self.hparams['loss_fn'])()

        self.train_step_outputs = []
        self.train_step_y = []
        self.val_step_outputs = []
        self.val_step_y = []
        self.test_step_outputs = []
        self.test_step_y = []


        self.save_hyperparameters()
    
    def custom_histogram_adder(self):
        for name,params in self.named_parameters():       
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def loggining_train(self):
        self.train_step_outputs = torch.concat(self.train_step_outputs).cpu()
        self.train_step_outputs = [ctc_decoder(x) for x in self.train_step_outputs]
        self.train_step_y = torch.concat(self.train_step_y).cpu()
        self.train_step_y = [x[x != 0] for x in self.train_step_y]
        
        divs = torch.LongTensor([textdistance.levenshtein.distance(s1.cpu().tolist(),s2[s2 != 0].cpu().tolist()) for s1,s2 in zip(self.train_step_outputs,self.train_step_y)])
        mean_div = divs.float().mean()
        wer = (divs == 0).float().mean()
        self.log('Mean div/Train', mean_div)
        self.log('Percent of equals words/Train', wer)
        self.train_step_outputs = []
        self.train_step_y = []

    def training_step(self, batch, batch_idx):
        x, y, lengths_y = batch
        output = self(x)
        self.train_step_outputs.append(output.detach().clone())
        self.train_step_y.append(y)        
        output = output.permute(1,0,2).log_softmax(2)

        output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long)

        loss = self.loss_fn(output, y, output_lengths, lengths_y)
        self.log("Loss/Train", loss)
        return loss

    def on_train_epoch_end(self):
        if(self.current_epoch==1):
            self.logger.experiment.add_graph(self, torch.rand(1,3,90,280, device=self.device))
        self.loggining_train()
        self.custom_histogram_adder()

    def loggining_val(self):
        self.val_step_outputs = torch.concat(self.val_step_outputs).cpu()
        self.val_step_outputs = [ctc_decoder(x) for x in self.val_step_outputs]
        self.val_step_y = torch.concat(self.val_step_y).cpu()
        self.val_step_y = [x[x != 0] for x in self.val_step_y]

        divs = torch.LongTensor([textdistance.levenshtein.distance(s1.tolist(),s2.tolist()) for s1,s2 in zip(self.val_step_outputs,self.val_step_y)])
        mean_div = divs.float().mean()
        wer = (divs == 0).float().mean()
        self.log("hp_metric", wer)
        self.log('Mean div/Val', mean_div)
        self.log('Percent of equals words/Val', wer)
        self.val_step_outputs = []
        self.val_step_y = []

    def logging_images(self, x, y, output, text_log):

        fig = plt.figure(tight_layout=True)
        for i in range(8):
            ax = fig.add_subplot(4, 2, i + 1)
            
            image = x[i]
            image = image * torch.Tensor([0.1414, 0.1284, 0.1476]).reshape(3,1,1).repeat(1,image.shape[1], image.shape[2]) * 255.0 + \
                255.0 * torch.Tensor([0.9076, 0.9098, 0.9088]).reshape(3,1,1).repeat(1,image.shape[1], image.shape[2])
            image = image.permute(1,2,0).numpy().astype('uint8')
            ax.imshow(image)

            text = ctc_decoder(output[i], self.char_set_pred)
            if text == '':
                text = 'None'
            ax.set_title(text)

        self.logger.experiment.add_figure(text_log, fig, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y, lengths_y = batch
        x_copy = x.cpu().detach().clone()

        output = self(x)
        if batch_idx == 0:
            self.logging_images(x_copy, y.cpu().detach().clone(), output.cpu().detach().clone(), 'Val')

        self.val_step_outputs.append(output.detach().clone())
        self.val_step_y.append(y)

        output = output.permute(1,0,2).log_softmax(2)
    
        output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long)
        loss = self.loss_fn(output, y, output_lengths, lengths_y)


        self.log("Loss/Val", loss)

    def on_validation_epoch_end(self):
        self.loggining_val()
        self.log('Learning rate', self.scheduler.get_lr()[0])

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.hparams['gamma'])
        return {'optimizer': self.optimizer, 'lr_scheduler':self.scheduler}

    def forward(self, x):
        x = self.encoder(x)
        # (B,C,H,W)
        x = x.permute(0,3,1,2)
        # (B,W,C,H)
        x = x.view(x.size(0),x.size(1),-1)
        # (B,W,C*H)
        x = self.fc_1(x)
        # (B,W,Emb)
        x = self.transformer_encoder(x)
        # (B,W,Emb)
        x = self.fc_2(x)
        # (B,W,ClassCount)
        return x

