import logging
import logging.config
import torch
import os
import numpy as np
import random
import time
import torch.nn as nn
import sys
from torch.utils.data import DataLoader,TensorDataset
from datetime import datetime
# from google.colab import drive

from torch.utils.tensorboard import SummaryWriter
from utils.args_utils import parse_args
from utils.data_utils import get_acdc,convert_masks
from utils.model import FCT


logging.config.fileConfig('./config/log_config.conf')
logger = logging.getLogger('mylog')

args = parse_args()
begin_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
assert torch.cuda.is_available() == True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            patience=10,
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


def main():
    # random seed
    setup_seed(args.random_seed)

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
    acdc_data[1] = convert_masks(acdc_data[1])
    acdc_data[0] = np.transpose(acdc_data[0], (0, 3, 1, 2)) # for the channels
    acdc_data[1] = np.transpose(acdc_data[1], (0, 3, 1, 2)) # for the channels
    acdc_data[0] = torch.Tensor(acdc_data[0]) # convert to tensors
    acdc_data[1] = torch.Tensor(acdc_data[1]) # convert to tensors
    acdc_data = TensorDataset(acdc_data[0], acdc_data[1])
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.decay)
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
    for epoch in range(first_epoch,args.max_epoch):
        model.train()
        # mini batch train
        train_loss_list = []
        grads_dict = {}
        abs_grads_dict = {}
        for i,(x,y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            pred_y = model(x)
            loss = loss_fn(pred_y[2],y)
            train_loss_list.append(loss)
            loss.backward()
            optimizer.step()
            for name, params in model.named_parameters():
                if name not in grads_dict:
                    grads_dict[name] = []
                    abs_grads_dict[name] = []
                if params.grad is not None:
                    grads_dict[name].append(params.grad.mean())
                    abs_grads_dict[name].append(params.grad.abs().mean())
        
        optimizer.zero_grad()
        train_loss = torch.tensor(train_loss_list).mean()
        if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss)
        if isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()
        # validate
        model.eval()
        validate_loss_list = []
        with torch.no_grad():
            for i,(x,y) in enumerate(validation_dataloader):
                x = x.to(device)
                y = y.to(device)
                pred_y = model(x)
                loss = loss_fn(pred_y[2],y)
                validate_loss_list.append(loss)
        validate_loss = torch.tensor(validate_loss_list).mean()
        # tensorboard
        tb_writer.add_scalar('loss/train_loss', train_loss, epoch)
        tb_writer.add_scalar('loss/validate_loss', validate_loss, epoch)
        tb_writer.add_scalar(f'lr/{epoch // 100}', optimizer.param_groups[0]["lr"], epoch)
        for name in grads_dict:
            # 分段看
            tb_writer.add_scalar(f'{name}/{epoch // 100}',torch.tensor(grads_dict[name]).mean(), epoch)
            tb_writer.add_scalar(f'abs_{name}/{epoch // 100}',torch.tensor(abs_grads_dict[name]).mean(), epoch)
        # log
        logger.info('-' * 35)
        logger.info(
            f'epoch: {epoch},train loss:{train_loss:.6f},validate loss:{validate_loss:.6f}'
        )
        logger.info('-' * 35)

        # model save
        if validate_loss < min_loss:
            min_loss = validate_loss
            torch.save(model.state_dict(), f'{args.checkpoint}/model/model.pt')
            # torch.jit.script(model).save(f'{args.checkpoint}/model/model_jit.pt')
            if args.colab:
                torch.save(model.state_dict(), f'/content/drive/MyDrive/model.pt')
                # torch.jit.script(model).save(f'/content/drive/MyDrive/model_jit.pt')
        
        # checkpoint 
        if epoch % 100 == 1:
            save_checkpoint('/content/drive/MyDrive/checkpoint',epoch,model,optimizer)


if __name__ == '__main__':
    main()