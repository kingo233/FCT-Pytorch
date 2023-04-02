import torch
import torchvision.transforms as transforms
import numpy as np
import random
import PIL.Image
import cv2
from torch.utils.data import Dataset
from .data_utils import convert_mask_single

class ACDCTrainDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=20,translate=(0.2,0.2),interpolation=transforms.InterpolationMode.BILINEAR)
        ])
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        seed = np.random.randint(2147483647)

        x = PIL.Image.fromarray(self.x[index].reshape(224,224))
        y = PIL.Image.fromarray(self.y[index].reshape(224,224))

        torch.manual_seed(seed)
        tar_x = np.array(self.transform(x))
        # cv2.imwrite('x.jpg',tar_x)
        tar_x = tar_x.reshape(1,224,224)

        torch.manual_seed(seed)
        tar_y = np.array(self.transform(y))
        tar_y = convert_mask_single(tar_y)
        # cv2.imwrite('y_0.jpg',tar_y[0]* 255)
        # cv2.imwrite('y_1.jpg',tar_y[1]* 255)
        # cv2.imwrite('y_2.jpg',tar_y[2]* 255)
        # cv2.imwrite('y_3.jpg',tar_y[3]* 255)


        torch.manual_seed(0)
        return torch.tensor(tar_x).float(),torch.tensor(tar_y).float()