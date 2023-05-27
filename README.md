# FCT-Pytorch
Pytorch implementation for The Fully Convolutional Transformer(FCT) 

## note
This repo can:
1. reproduces the origin aurhor's work on tensorflow.You need reference the original repo's issue that they only use ACDC train set(split ACDC/traning set into 7:2:1 train:validation:test).You can get dice 92.9

2. Get about 90 dice on official test set if your train on the whole train set(using ACDC/training and test on ACDC/testing).


## training
1. Get ACDC dataset.And remember to delete `.md` file in your ACDC dataset folder
2. use `python main.py` to start training
