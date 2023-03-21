import torch

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

import string
import albumentations.pytorch
import albumentations as A
import cv2

from model import CaptchaDataset, CaptchaModel


hparams = {
    'learning_rate': 0.0008,
    'gamma': 0.98,
    'max_length_output': 16,
    'max_count_dataset_train': 5000,
    'max_count_dataset_val': 500,
    'loss_fn': 'CTCLoss',
    'batch_count_train' : 32,
    'batch_count_val' : 32,
    'global_seed': 42
    }

seed_everything(hparams['global_seed'])

char_set_pred = ' ' + string.digits + string.ascii_lowercase

train_dir = 'images/train'
val_dir = 'images/val'
test_dir = 'images/test'

train_transform = A.Compose([
    # A.GaussNoise((0,100)),
    # A.RandomBrightnessContrast(),
    # A.RandomGamma(),
    # A.CLAHE(),
    # A.Blur(),
    # A.ShiftScaleRotate(),
    # A.RGBShift(),
    A.ElasticTransform(alpha=0.1,sigma=1,alpha_affine=5, border_mode=cv2.BORDER_WRAP),
    A.Normalize((0.9076, 0.9098, 0.9088), (0.1414, 0.1284, 0.1476)),
    A.pytorch.transforms.ToTensorV2(),
],p=1)

val_transform = A.Compose([
    A.Normalize((0.9076, 0.9098, 0.9088), (0.1414, 0.1284, 0.1476)),
    A.pytorch.transforms.ToTensorV2(),
], p =1)

dataset_train = CaptchaDataset(train_dir, 'train', char_set_pred, transform=train_transform, hparams=hparams)
dataset_val = CaptchaDataset(val_dir, 'val', char_set_pred, transform=val_transform, hparams=hparams)

captcha_model = CaptchaModel(hparams, char_set_pred)
# model = captcha_model.load_from_checkpoint(r'tb_logs\model_ctc\version_40\checkpoints\epoch=26-step=6912.ckpt')

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=False)


logger = TensorBoardLogger('tb_logs', 'model_ctc')

trainer = pl.Trainer(max_epochs=500, accelerator="gpu", logger=logger)
trainer.fit(model=captcha_model, train_dataloaders=train_loader,
            val_dataloaders=val_loader)