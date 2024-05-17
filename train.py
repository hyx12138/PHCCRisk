import os
import time
import numpy as np
import pandas as pd
import torch
import monai
import matplotlib.pyplot as plt
from monai.optimizers import WarmupCosineSchedule
from monai.data import (
    CacheDataset,
    PersistentDataset,
    DataLoader,
    set_track_meta,
)
from transformation import ResNet3D_transformations
from ResNet3D_Builder import ResNet3D
from Loss import AsymmetricLossOptimized
from monai.utils import set_determinism
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score,  precision_score,  recall_score, precision_recall_curve, roc_curve
from torch.utils.tensorboard import SummaryWriter
import argparse

def get_data_dict(dataset_path):
    df = pd.read_csv(dataset_path)

    ID_list = []
    AP_I_list = []
    AP_S_list = []
    VP_I_list = []
    VP_S_list = []
    labels_list = []

    AP_I_list = df['AP_image'].to_list()
    AP_S_list = df['AP_seg'].to_list()
    VP_I_list = df['VP_image'].to_list()
    VP_S_list = df['VP_seg'].to_list()
    ID_list = df['ID'].to_list()
    labels_list = df['Label'].to_list()

    data_dicts = [{"AP_image": a, 'AP_seg': b, 'VP_image': c, "VP_seg": d, "Label": e, 'ID': i}
                  for a,b,c,d,e, i in zip(AP_I_list, AP_S_list, VP_I_list, VP_S_list, labels_list, ID_list)]

    return data_dicts

def setup_seed(seed = 3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_determinism(seed=seed)

def main_worker(args):
    device = torch.device(f"cuda:0")
                                       
    writer = SummaryWriter(f'./PHCCRisk/runs/{args.sequence}/')

    model = ResNet3D(
        monai.networks.nets.__dict__[args.arch],
        spatial_dims = 3,
        n_input_channels = 1,
        num_classes = 1, 
        pretrained_dict_path = args.pretrain_path
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.wd)
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay=args.wd, momentum=args.momentum)
    model = model.to(device)

    train_dataset_path = f'Path_to_train_dataset'
    val_dataset_path = f'Path_to_val_dataset'

    save_dir = f'./PHCCRisk/results/{args.sequence}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cache_dir = f'./PHCCRisk/monai_cache/{args.sequence}/'

    max_epochs = args.epochs
    warmup_epochs = args.warm

    monai_start = time.time()

    train_trans, val_trans = ResNet3D_transformations(flag = args.sequence)


    train_dicts = get_data_dict(train_dataset_path)
    val_dicts = get_data_dict(val_dataset_path)

    train_ds = PersistentDataset(data=train_dicts,
                                transform=train_trans,
                                cache_dir=cache_dir,
    )

    val_ds = PersistentDataset(data=val_dicts,
                            transform=val_trans,
                            cache_dir=cache_dir,
    ) 

    train_loader = DataLoader(train_ds, num_workers=args.workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False, pin_memory=True)

    loss_function = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)

    scaler = torch.cuda.amp.GradScaler()

    set_track_meta(False)

    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_epochs, t_total=max_epochs, cycles=0.5, last_epoch=-1)
    metrics = train_process(
        model, optimizer, scaler, loss_function, train_loader, val_loader, scheduler, writer, save_dir=save_dir, max_epochs=max_epochs, device=device, args=args
        )

    m_total_time = time.time() - monai_start
    print(
        f"total time of training:{m_total_time:.4f}"
    )
    writer.close()

def train_process(model, optimizer, scaler, loss_function, train_loader, val_loader, scheduler, writer, save_dir, max_epochs, device, args):
    val_interval = args.val_freq  # do validation for every epoch

    epoch_times = []
    total_start = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")

        epoch_first_lr = scheduler.get_last_lr()
    
        model.train()
        epoch_loss = 0.
        train_loader_iterator = iter(train_loader)

        for step in range(1, len(train_loader) + 1):
            batch_data = next(train_loader_iterator)
            inputs, labels = ( # [b,c,w,h,d]
                    batch_data[f"{args.sequence}_image"].permute(0,1,4,3,2).to(device), # [b,c,d,h,w]
                    batch_data["Label"].to(device),
                    )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs.squeeze(0,2), labels.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
        scheduler.step()

        epoch_loss /= step
        print(f"epoch {epoch + 1}---average loss: {epoch_loss:.4f}---begin_lr: {epoch_first_lr}")
        writer.add_scalar('lr', epoch_first_lr[0], epoch + 1)
        writer.add_scalar('train loss', epoch_loss, epoch + 1)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.
            y_true = []
            y_prob = []
            with torch.no_grad():
                val_loader_iterator = iter(val_loader)

                for _ in range(len(val_loader)):
                    val_data = next(val_loader_iterator)
                    inputs_val, labels_val = ( # [b,c,d,h,w]
                                    val_data[f"{args.sequence}_image"].to(device), # [b,c,d,h,w]
                                    val_data["Label"].to(device),
                    )
                    with torch.cuda.amp.autocast():
                        val_outputs = model(inputs_val)
                        loss = loss_function(val_outputs, labels.float())

                    val_loss += loss.item()

                    y_true.append(labels_val.item())
                    y_prob.append(val_outputs.item())
                val_loss /= len(val_loader)

                AUC = roc_auc_score(y_true, y_prob)

                print(
                    f"current epoch: {epoch + 1}"
                    f" val auc: {AUC:.4f}\n"
                )
        writer.add_scalar('val loss', val_loss, epoch + 1)
        writer.add_scalar('val AUC', AUC, epoch + 1)

        print(
            f"time consuming of epoch {epoch + 1} is:"
            f" {(time.time() - epoch_start):.4f}"
        )
        epoch_times.append(time.time() - epoch_start)


    total_time = time.time() - total_start
    print("-" * 30 + 
        f"train completed"
        f" total time: {total_time:.4f}" + "-" * 30
    )

def main():
    args = parser.parse_args()
    setup_seed(args.seed)
    main_worker(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet3D Training")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--epochs", default=50, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--warm", default=5, type=int, metavar="N", help="number of total epochs to warmup"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        metavar="N",
        help="mini-batch size",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=3.5e-5,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
    )
    parser.add_argument(
        "--wd",
        "--w_decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay",
        dest="wd",
    )
    parser.add_argument(
        "-p",
        "--val_freq",
        default=1,
        type=int,
        metavar="N",
        help="print frequency",
    )
    parser.add_argument(
        "--sequence",
        default='AP',
        type=str,
        help="sequence to use",
    )
    parser.add_argument(
        "--seed", default=3407, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--pretrain_path", type=str, help="path to pretrain weights"
    )
    main()

'''
python3 ./PHCCRisk/train.py \
        -a resnet18 --workers 4 --epochs 50 --warm 5 --batch-size 16 \
        --seed 3407 --lr 3e-5 --wd 1e-4 --sequence VP \
        --pretrain_path './PHCCRisk/resnet_18_23dataset.pth'
'''

