import os
import cv2
import nibabel as nib
import numpy as np
from monai import data, transforms
from monai.data import load_decathlon_datalist

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, 'dataset.json')
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],pixdim=[1.5,1.5,10]),
            transforms.NormalizeIntensityd(keys=["image", "label"]),
            # transforms.Resized(keys=['image', 'label'],spatial_size=[256,256,8],mode='trilinear')
        ]
    )
    
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
        ]
    )
    if args.predict_mode:
        pass
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "test", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        loader = [train_loader, val_loader]
    return loader

def get_acdc(path,input_size=(224,224,1)):
    """
    Read images and masks for the ACDC dataset
    """
    all_imgs = []
    all_gt = []
    all_header = []
    all_affine = []
    info = []
    for root, directories, files in os.walk(path):
        for file in files:
            if ".gz" and "frame" in file:
                if "_gt" not in file:
                    img_path = root + "/" + file
                    img = nib.load(img_path).get_fdata()
                    all_header.append(nib.load(img_path).header)
                    all_affine.append(nib.load(img_path).affine)                   
                    for idx in range(img.shape[2]):
                        i = cv2.resize(img[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)
                        all_imgs.append(i)
                        
                else:
                    img_path = root + "/" + file
                    img = nib.load(img_path).get_fdata()
                    for idx in range(img.shape[2]):
                        i = cv2.resize(img[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)
                        all_gt.append(i)
            

    data = [all_imgs, all_gt, info]                  
 
      
    data[0] = np.expand_dims(data[0], axis=3)
    if path[-9:] != "true_test":
        data[1] = np.expand_dims(data[1], axis=3)
    
    return data, all_affine, all_header

def convert_masks(y, data="acdc"):
    """
    Given one masks with many classes create one mask per class
    """
    
    if data == "acdc":
        # initialize
        masks = np.zeros((y.shape[0], y.shape[1], y.shape[2], 4))
        
        for i in range(y.shape[0]):
            masks[i][:,:,0] = np.where(y[i]==0, 1, 0)[:,:,-1] 
            masks[i][:,:,1] = np.where(y[i]==1, 1, 0)[:,:,-1] 
            masks[i][:,:,2] = np.where(y[i]==2, 1, 0)[:,:,-1] 
            masks[i][:,:,3] = np.where(y[i]==3, 1, 0)[:,:,-1]
            
    elif data == "synapse":
        masks = np.zeros((y.shape[0], y.shape[1], y.shape[2], 9))
        
        for i in range(y.shape[0]):
            masks[i][:,:,0] = np.where(y[i]==0, 1, 0)[:,:,-1]
            masks[i][:,:,1] = np.where(y[i]==1, 1, 0)[:,:,-1]  
            masks[i][:,:,2] = np.where(y[i]==2, 1, 0)[:,:,-1]  
            masks[i][:,:,3] = np.where(y[i]==3, 1, 0)[:,:,-1]  
            masks[i][:,:,4] = np.where(y[i]==4, 1, 0)[:,:,-1]  
            masks[i][:,:,5] = np.where(y[i]==5, 1, 0)[:,:,-1]  
            masks[i][:,:,6] = np.where(y[i]==6, 1, 0)[:,:,-1]  
            masks[i][:,:,7] = np.where(y[i]==7, 1, 0)[:,:,-1]  
            masks[i][:,:,8] = np.where(y[i]==8, 1, 0)[:,:,-1]  
            
    else:
        print("Data set not recognized")
        
    return masks

def convert_mask_single(y):
    """
    Given one masks with many classes create one mask per class
    y: shape (w,h)
    """
    mask = np.zeros((4, y.shape[0], y.shape[1]))
    mask[0, :, :] = np.where(y == 0, 1, 0)
    mask[1, :, :] = np.where(y == 1, 1, 0)
    mask[2, :, :] = np.where(y == 2, 1, 0)
    mask[3, :, :] = np.where(y == 3, 1, 0)

    return mask

def visualize(image_raw,mask):
    """
    iamge_raw:gray image with shape [width,height,1]
    mask: segment mask image with shape [num_class,width,height]
    this function return an image using multi color to visualize masks in raw image
    """
    # Convert grayscale image to RGB
    image = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB)
    
    # Get the number of classes (i.e. channels) in the mask
    num_class = mask.shape[0]
    
    # Define colors for each class (using a simple color map)
    colors = []
    for i in range(1, num_class):  # skip first class (background)
        hue = int(i/float(num_class-1) * 179)
        color = np.zeros((1, 1, 3), dtype=np.uint8)
        color[0, 0, 0] = hue
        color[0, 0, 1:] = 255
        color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
        colors.append(color)

    # Overlay each non-background class mask with a different color on the original image
    for i in range(1, num_class):
        class_mask = mask[i, :, :]
        class_mask = np.repeat(class_mask[:, :, np.newaxis], 3, axis=2)
        class_mask = class_mask.astype(image.dtype)
        class_mask = class_mask * colors[i-1]
        image = cv2.addWeighted(image, 1.0, class_mask, 0.5, 0.0)

    return image