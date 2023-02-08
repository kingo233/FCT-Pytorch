import logging
import logging.config
import torch
import os
import numpy as np
import random
import time
import torch.nn as nn
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from utils.args_utils import parse_args
from utils.data_utils import get_loader


logging.config.fileConfig('./config/log_config.conf')
logger = logging.getLogger('mylog')

args = parse_args()
begin_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

def main():
    # random seed
    setup_seed(args.random_seed)

    # make dirs
    make_dir(args)

    # logger
    logger.info(args)

    # tensorboard writer
    tb_writer = SummaryWriter(log_dir=os.path.join(args.checkpoint, 'runs'))

    # get data
    loader = get_loader(args)
    if len(loader) == 2:
        train_loader,test_loader = loader
    else:
        pass
    for idx,batch in enumerate(train_loader):
        print(batch)

if __name__ == '__main__':
    main()