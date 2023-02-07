import os
from monai import data, transforms
from monai.data import load_decathlon_datalist

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, 'dataset.json')
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
        ]
    )
    
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
        ]
    )
    if args.test_mode:
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