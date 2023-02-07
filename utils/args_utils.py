import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='FCT for medical image')

    # msic
    # https://arxiv.org/pdf/2109.08203.pdf  3407 or other you like
    parser.add_argument('--random_seed',default=0,type=int,help='random seed,default 0')
    parser.add_argument('--checkpoint', default='./output_model', type=str, help='checkpoint model')
    parser.add_argument('--data_dir',default='Task027_ACDC')
    parser.add_argument('--test_mode',default=False,help='Train or Predict')
    parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument("--workers", default=4, type=int, help="number of workers")
    
    # train parameters
    parser.add_argument('--batch_size',default=1,type=int,help='batch size,default size 1')
    parser.add_argument('--max_epoch',default=100,type=int,help='the max number with epoch')
    parser.add_argument('--lr',default=5e-7,help='learning rate,when resume,set to 0 means using checkpoint lr,or a new lr')
    parser.add_argument('--decay',default=0.001,help='L2 norm')
    parser.add_argument('--lr_factor',default=0.5,help='dynamic learning rate factor,when loss not change,new lr = old_lr * lr_factor')
    parser.add_argument('--min_lr',default=5e-7,help='min dynamic learing rate')
    parser.add_argument('--lr_scheduler',default='ReduceLROnPlateau')
    parser.add_argument('--resume',action='store_true')

    # network parameters
    

    args = parser.parse_args()

    return args
