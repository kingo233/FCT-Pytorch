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