import logging
import logging.config
import torch
import os
import numpy as np
import random
import time
import torch.nn as nn
import sys
import monai.metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from datetime import datetime
# from google.colab import drive

from torch.utils.tensorboard import SummaryWriter
from utils.args_utils import parse_args
from utils.data_utils import get_acdc,convert_masks
from utils.model import FCT
from utils.dataset_utils import ACDCTrainDataset


logging.config.fileConfig('./config/log_config.conf')
logger = logging.getLogger('mylog')

args = parse_args()
begin_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'mps'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_dir(args):
    args.checkpoint = os.path.join(args.checkpoint, begin_time)
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    if not os.path.exists(os.path.join(args.checkpoint, 'model')):
        os.makedirs(os.path.join(args.checkpoint, 'model'))
    if not os.path.exists(os.path.join(args.checkpoint, 'runs')):
        os.makedirs(os.path.join(args.checkpoint, 'runs'))


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

def save_checkpoint(path: str,
                    epoch: int,
                    model: nn.Module,
                    optimizer,
                    safe_replacement: bool = True):
    """
    Save a checkpoint of the current state of the training, so it can be resumed.
    This checkpointing function assumes that there are no learning rate schedulers or gradient scalers for automatic
    mixed precision.
    :param path:
        Path for your checkpoint file
    :param epoch:
        Current (completed) epoch
    :param model:
        nn.Module containing the model 
    :param optimizers:
        Optimizer 
    :param safe_replacement:
        Keep old checkpoint until the new one has been completed
    :return:
    """
 
    # Data dictionary to be saved
    data = {
        'epoch': epoch,
        # Current time (UNIX timestamp)
        'time': time.time(),
        # State dict for all the modules
        'model': model.state_dict(),
        # State dict for all the optimizers
        'optimizer': optimizer.state_dict()
    }

    # Safe replacement of old checkpoint
    temp_file = None
    if os.path.exists(path) and safe_replacement:
        # There's an old checkpoint. Rename it!
        temp_file = path + '.old'
        os.rename(path, temp_file)

    # Save the new checkpoint
    with open(path, 'wb') as fp:
        torch.save(data, fp)
        # Flush and sync the FS
        fp.flush()
        os.fsync(fp.fileno())

    # Remove the old checkpoint
    if temp_file is not None:
        os.unlink(path + '.old')


def load_checkpoint(path: str,
                    default_epoch: int,
                    model: nn.Module,
                    optimizer,
                    verbose: bool = True):
    """
    Try to load a checkpoint to resume the training.
    :param path:
        Path for your checkpoint file
    :param default_epoch:
        Initial value for "epoch" (in case there are not snapshots)
    :param modules:
        nn.Module containing the model or a list of nn.Module objects. They are assumed to stay on the same device
    :param optimizers:
        Optimizer or list of optimizers
    :param verbose:
        Verbose mode
    :return:
        Next epoch
    """

    # If there's a checkpoint
    if os.path.exists(path):
        # Load data
        data = torch.load(path)

        # Inform the user that we are loading the checkpoint
        if verbose:
            print(f"Loaded checkpoint saved at {datetime.fromtimestamp(data['time']).strftime('%Y-%m-%d %H:%M:%S')}. "
                  f"Resuming from epoch {data['epoch']}")

        model.load_state_dict(data['model'])
        optimizer.load_state_dict(data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if hasattr(v,'cuda'):
                    state[k] = v.cuda()

        # Next epoch
        return data['epoch'] + 1
    else:
        return default_epoch

def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def compute_dice(pred_y, y):
    """
    Computes the Dice coefficient for each class in the ACDC dataset.
    Assumes binary masks with shape (num_masks, num_classes, height, width).
    """
    epsilon = 1e-6
    num_masks = pred_y.shape[0]
    num_classes = pred_y.shape[1]
    dice_scores = torch.zeros((num_classes,)).to(device)
    
    for c in range(num_classes):
        intersection = torch.sum(pred_y[:,c] * y[:,c])
        sum_masks = torch.sum(pred_y[:,c]) + torch.sum(y[:,c])
        dice_scores[c] = (2. * intersection + epsilon) / (sum_masks + epsilon)
    
    return dice_scores


def main():
    # random seed
    # setup_seed(args.random_seed)

    # make dirs
    make_dir(args)

    # logger
    logger.info(args)

    # tensorboard writer
    tb_writer = SummaryWriter(log_dir=os.path.join(args.checkpoint, 'runs'))

    # model instatation
    model = FCT(args)
    model.apply(init_weights)

    # get data
    # training
    acdc_data, _, _ = get_acdc('ACDC/training')
    # acdc_data[1] = convert_masks(acdc_data[1])
    # acdc_data[0] = np.transpose(acdc_data[0], (0, 3, 1, 2)) # for the channels
    # acdc_data[1] = np.transpose(acdc_data[1], (0, 3, 1, 2)) # for the channels
    # acdc_data[0] = torch.Tensor(acdc_data[0])# convert to tensors
    # acdc_data[1] = torch.Tensor(acdc_data[1])# convert to tensors
    acdc_data = ACDCTrainDataset(acdc_data[0], acdc_data[1])
    train_dataloader = DataLoader(acdc_data, batch_size=args.batch_size)
    # validation
    acdc_data, _, _ = get_acdc('ACDC/testing')
    acdc_data[1] = convert_masks(acdc_data[1])
    acdc_data[0] = np.transpose(acdc_data[0], (0, 3, 1, 2)) # for the channels
    acdc_data[1] = np.transpose(acdc_data[1], (0, 3, 1, 2)) # for the channels
    acdc_data[0] = torch.Tensor(acdc_data[0]) # convert to tensors
    acdc_data[1] = torch.Tensor(acdc_data[1]) # convert to tensors
    acdc_data = TensorDataset(acdc_data[0], acdc_data[1])
    validation_dataloader = DataLoader(acdc_data, batch_size=args.batch_size)

    # initialize the loss function
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)# , weight_decay=args.decay)
    scheduler = get_lr_scheduler(args,optimizer)
    model.to(device)

    # resume
    # TODO
    first_epoch = 0
    if args.resume:
        first_epoch = load_checkpoint('checkpoint', 0, model, optimizer)
        for g in optimizer.param_groups:
            g['lr'] = args.lr
            g['weight_decay'] = args.decay

    min_loss = sys.maxsize
    train_step = 0
    for epoch in range(first_epoch,args.max_epoch):
        model.train()
        # mini batch train
        train_loss_list = []
        grads_dict = {}
        abs_grads_dict = {}
        train_mean_dice_list = []
        train_LV_dice_list = []
        train_RV_dice_list = []
        train_MYO_dice_list = []
        for i,(x,y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            pred_y = model(x)
            # dsn
            down1 = F.interpolate(y,112)
            down2 = F.interpolate(y,56)
            loss = (loss_fn(pred_y[2],y) * 0.57 + loss_fn(pred_y[1],down1) * 0.29 + loss_fn(pred_y[0],down2) * 0.14)
            # loss *= 1e4 # scale grad
            # with torch.no_grad():
            #     loss /= 1e4
            
            train_loss_list.append(loss)
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()

            # train dice
            y_pred = torch.argmax(pred_y[2],axis=1)
            y_pred_onehot = torch.nn.functional.one_hot(y_pred,4).permute(0,3,1,2)
            dice = compute_dice(y_pred_onehot,y)
            dice_LV = dice[3]; train_LV_dice_list.append(dice_LV)
            dice_RV = dice[1]; train_RV_dice_list.append(dice_RV)
            dice_MYO = dice[2]; train_MYO_dice_list.append(dice_MYO)
            train_mean_dice_list.append(dice[1:].mean())
            # save grad
            for name, params in model.named_parameters():
                if name not in grads_dict:
                    grads_dict[name] = []
                    abs_grads_dict[name] = []
                if params.grad is not None:
                    grads_dict[name].append(params.grad.mean())
                    abs_grads_dict[name].append(params.grad.abs().mean())
                    tb_writer.add_scalar(f'batch_abs_{name}',params.grad.abs().mean(), train_step)
            train_step += 1
        
        train_loss = torch.tensor(train_loss_list).mean()
        # validate
        model.eval()
        validate_loss_list = []
        mean_dice_list = []
        LV_dice_list = []
        RV_dice_list = []
        MYO_dice_list = []
        with torch.no_grad():
            for i,(x,y) in enumerate(validation_dataloader):
                x = x.to(device)
                y = y.to(device)
                pred_y = model(x)

                down1 = F.interpolate(y,112)
                down2 = F.interpolate(y,56)
                loss = loss_fn(pred_y[2],y) * 0.57 + loss_fn(pred_y[1],down1) * 0.29 + loss_fn(pred_y[0],down2) * 0.14
                validate_loss_list.append(loss)

                # validate dice
                y_pred = torch.argmax(pred_y[2],axis=1)
                y_pred_onehot = torch.nn.functional.one_hot(y_pred,4).permute(0,3,1,2)
                dice = compute_dice(y_pred_onehot,y)
                dice_LV = dice[3]; LV_dice_list.append(dice_LV)
                dice_RV = dice[1]; RV_dice_list.append(dice_RV)
                dice_MYO = dice[2]; MYO_dice_list.append(dice_MYO)
                mean_dice_list.append(dice[1:].mean())
                
        validate_loss = torch.tensor(validate_loss_list).mean()

        # lr scheduler
        if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(validate_loss)
        if isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()

        # train dice calc
        train_dice_coef = torch.tensor(train_mean_dice_list).mean()
        train_LV_dice = torch.tensor(train_LV_dice_list).mean()
        train_RV_dice = torch.tensor(train_RV_dice_list).mean()
        train_MYO_dice = torch.tensor(train_MYO_dice_list).mean()

        # validate dice calc
        dice_coef = torch.tensor(mean_dice_list).mean()
        LV_dice = torch.tensor(LV_dice_list).mean()
        RV_dice = torch.tensor(RV_dice_list).mean()
        MYO_dice = torch.tensor(MYO_dice_list).mean()
        # tensorboard
        tb_writer.add_scalar('loss/train_loss', train_loss, epoch)
        tb_writer.add_scalar('loss/validate_loss', validate_loss, epoch)
        tb_writer.add_scalar('dice/all_validate_dice',dice_coef, epoch)
        tb_writer.add_scalar('dice/LV_dice',LV_dice, epoch)
        tb_writer.add_scalar('dice/RV_dice',RV_dice, epoch)
        tb_writer.add_scalar('dice/MYO_dice',MYO_dice, epoch)
        tb_writer.add_scalar('dice/all_train_dice',train_dice_coef, epoch)
        tb_writer.add_scalar('dice/train_LV_dice',train_LV_dice, epoch)
        tb_writer.add_scalar('dice/train_RV_dice',train_RV_dice, epoch)
        tb_writer.add_scalar('dice/train_MYO_dice',train_MYO_dice, epoch)

        tb_writer.add_scalar(f'lr/{epoch // 100}', optimizer.param_groups[0]["lr"], epoch)
        for name in grads_dict:
            # 分段看
            tb_writer.add_scalar(f'{name}/{epoch // 100}',torch.tensor(grads_dict[name]).mean(), epoch)
            tb_writer.add_scalar(f'abs_{name}/{epoch // 100}',torch.tensor(abs_grads_dict[name]).mean(), epoch)
        # log
        logger.info('-' * 35)
        logger.info(
            f'epoch: {epoch},train loss:{train_loss:.6f},validate loss:{validate_loss:.6f},LV_dice:{LV_dice},RV_dice:{RV_dice},MYO_dice:{MYO_dice}'
        )
        logger.info('-' * 35)

        # model save
        if validate_loss < min_loss:
            min_loss = validate_loss
            torch.save(model.state_dict(), f'{args.checkpoint}/model/fct.pt')
            # torch.jit.script(model).save(f'{args.checkpoint}/model/model_jit.pt')
            if args.colab:
                torch.save(model.state_dict(), f'/content/drive/MyDrive/fct.pt')
                # torch.jit.script(model).save(f'/content/drive/MyDrive/model_jit.pt')
        
        # checkpoint 
        if epoch % 5 == 0:
            save_checkpoint('checkpoint',epoch,model,optimizer)


if __name__ == '__main__':
    main()