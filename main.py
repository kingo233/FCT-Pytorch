import torch
import numpy as np
import torch.nn as nn
import monai.metrics
import lightning as L
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from datetime import datetime
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
# from google.colab import drive

from torch.utils.tensorboard import SummaryWriter
from utils.args_utils import parse_args
from utils.data_utils import get_acdc,convert_masks
from utils.model import FCT
from utils.dataset_utils import ACDCTrainDataset

args = parse_args()

def get_lr_scheduler(args,optimizer):
    if args.lr_scheduler == 'none':
        return None
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            verbose=True,
            threshold=1e-6,
            patience=5,
            min_lr=args.min_lr)
        return scheduler
    if args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=500
        )
        return scheduler

@torch.no_grad()
def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def main():
    # model instatation
    model = FCT(args)
    model.apply(init_weights)

    # get data
    # training
    acdc_data, _, _ = get_acdc('ACDC/training', input_size=(args.img_size,args.img_size,1))
    acdc_data = ACDCTrainDataset(acdc_data[0], acdc_data[1],args)
    train_dataloader = DataLoader(acdc_data, batch_size=args.batch_size,num_workers=args.workers)
    # validation
    acdc_data, _, _ = get_acdc('ACDC/testing', input_size=(args.img_size,args.img_size,1))
    acdc_data[1] = convert_masks(acdc_data[1])
    acdc_data[0] = np.transpose(acdc_data[0], (0, 3, 1, 2)) # for the channels
    acdc_data[1] = np.transpose(acdc_data[1], (0, 3, 1, 2)) # for the channels
    acdc_data[0] = torch.Tensor(acdc_data[0]) # convert to tensors
    acdc_data[1] = torch.Tensor(acdc_data[1]) # convert to tensors
    acdc_data = TensorDataset(acdc_data[0], acdc_data[1])
    validation_dataloader = DataLoader(acdc_data, batch_size=args.batch_size,num_workers=args.workers)

    # resume
    # TODO need debug
    if args.resume:
        if args.new_param:
            model = FCT.load_from_checkpoint('lightning_logs/version_2/checkpoints/epoch=74-step=4500.ckpt',args=args)
        else:
            # load weights,old hyper parameter and optimizer state 
            model = FCT.load_from_checkpoint('this is path')
    
    precision = '16-mixed' if args.amp else 32
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(precision=precision,max_epochs=args.max_epoch,callbacks=[lr_monitor])
    trainer.fit(model=model,train_dataloaders=train_dataloader,val_dataloaders=validation_dataloader)


if __name__ == '__main__':
    main()