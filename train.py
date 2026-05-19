
from html import parser
import sys
import os
import torch
import argparse
import time
import numpy as np
import torch.nn.functional as F
from utils.auxil import *
from torch_geometric.loader import DataLoader

import math
import gc
from utils.DataLoader import WDNDataset
from utils.early_stopping import EarlyStopping
from models.ConfigModels import select_model
 
torch.cuda.empty_cache()
gc.collect()

"""
Configurations for training and evaluation of GNNs on WDNs
"""
_MODEL             = "gatres_small"
_LR                = 0.0005
_WEIGHT_DECAY      = 0.000001
_EPOCHS            = 250
_MASK_RATE         = 0.95
_DATASET_PATHS = ["/content/datasets/ctown.zip"]
_INPUT_PATHS   = ["/content/inputs/ctown.inp"]
_MODEL_NAME        = None
_BATCH_SIZE    = 8
_PATIENCE          = 250
_MIN_DELTA         = 1e-6
_DEVICE            = "cuda"
_SAVE_PATH     = "/content/drive/MyDrive/WDN/checkpoints/ctown_gatres"
_MODEL_PATH    = "/content/drive/MyDrive/WDN/checkpoints/ctown_gatres10k_12/best_GATResMeanConv_small_znorm_15b_32c_20260510_1654.pth"
_RWPE_STEPS         = 0
_LAPEV_K            = 0
_DO_LOAD            = False
_FEATURE             = "pressure"
_RANDOM_SEED = 42



def get_arguments(raw_args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=_MODEL,
        type=str,
        choices=["gatres_small", "gatres_small_rwpe", "gatres_small_signnet"],
        help="support model selection only.",
    )
    parser.add_argument("--lr", default=_LR, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=_WEIGHT_DECAY, type=float, help="weight decay")
    parser.add_argument("--epochs", default=_EPOCHS, type=int, help="number of epochs")
    parser.add_argument("--mask_rate", default=_MASK_RATE, type=float, help="masking ratio")
    parser.add_argument(
        "--dataset_paths",
        default=_DATASET_PATHS,
        type=str,
        nargs="*",
        action="store",
        help="list of dataset paths used for training and validation (order-sensitive)",
    )
    parser.add_argument(
        "--input_paths",
        default=_INPUT_PATHS,
        type=str,
        nargs="*",
        action="store",
        help="list of WDN input paths used for training and validation (order-sensitive)",
    )
    parser.add_argument("--model_name", default=_MODEL_NAME, type=str, help="Name of model. Keep its empty to use the name of class by default")
    parser.add_argument("--batch_size", default=_BATCH_SIZE, type=int, help="batch size")
    parser.add_argument(
        "--patience", default=_PATIENCE, type=int, help="Early stopping patience in these epochs. If val_loss unchanges, the training is stopped"
    )
    parser.add_argument("--min_delta", default=_MIN_DELTA, type=float, help="delta between last_loss and best_loss")
    parser.add_argument(
        "--device",
        default=_DEVICE,
        type=str,
        choices=["cuda", "cpu"],
        help="Training device. If gpu is unavailable, device is set to cpu. Support: cuda| cpu",
    )
    parser.add_argument("--save_path", default=_SAVE_PATH, type=str, help="Path to store model weights")
    parser.add_argument("--model_path", default=_MODEL_PATH, type=str, help="Model path")
    parser.add_argument("--rwpe_steps", default=_RWPE_STEPS, type=int, help="Number of Random Walk PE steps appended to node features. 0 = disabled.")
    parser.add_argument("--lapev_k", default=_LAPEV_K, type=int, help="Number of Laplacian eigenvectors for SignNet PE. 0 = disabled.")
    parser.add_argument("--do_load", default=_DO_LOAD, action="store_true", help="Whether to load model weights from model_path")
    parser.add_argument("--feature", default=_FEATURE, choices=["pressure", "head"], type=str, help="feature input")
    parser.add_argument("--seed", default=_RANDOM_SEED, type=int, help="Random seed for reproducibility")


    args = parser.parse_args(args=raw_args)
    return args




"""
 Main training loop 
"""

def get_default_datasets(args):
    rwpe_steps = getattr(args, "rwpe_steps", 0)
    lapev_k    = getattr(args, "lapev_k", 0)

    train_ds = WDNDataset(
        zip_file_paths=args.dataset_paths,
        input_paths=args.input_paths,
        feature=args.feature,
        from_set="train",
        mean=None, std=None,
        norm_type="znorm",
        rwpe_steps=rwpe_steps,
        lapev_k=lapev_k,
    )

    val_ds = WDNDataset(
        zip_file_paths=args.dataset_paths,
        input_paths=args.input_paths,
        feature=args.feature,
        from_set="valid",
        mean=train_ds.mean, std=train_ds.std,
        norm_type="znorm",
        rwpe_steps=rwpe_steps,
        lapev_k=lapev_k,
    )

    return train_ds, val_ds
 
 
def train_one_epoch(
    model, 
    optimizer, 
    loader, 
    mask_rate, 
    device,
    mean, 
    std, 
    criterion, 
    metric_fn_dict, 

):
    model.train()
    total_loss = 0
    total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
    record_metric_dict = {}
 
    for data in loader:
        optimizer.zero_grad()
        data.x           = data.x.to(device)
        data.y           = data.y.to(device)
        data.edge_index  = data.edge_index.to(device)
 
        num_nodes  = torch.unique(data.batch, return_counts=True)[1]
        batch_mask = generate_batch_mask(num_nodes=num_nodes, mask_rate=mask_rate, required_idx=[])
        data.x[batch_mask] = 0
        x_input = torch.cat([data.x, data.pe.to(device)], dim=-1) if hasattr(data, "pe") else data.x
        eig     = data.eig.to(device) if hasattr(data, "eig") else None
        out     = model(x_input, data.edge_index, eig)
 
        y_pred = out[batch_mask]
        y_true = data.y[batch_mask]
        y_pred_rescaled = descale(scaled_data=y_pred, mean=mean, std=std)
        y_true_rescaled = descale(scaled_data=y_true, mean=mean, std=std)
 
        tr_loss = criterion(y_pred, y_true)
        tr_loss.backward()
        optimizer.step()
 
        total_loss += float(tr_loss) * data.num_graphs
        for k, fn in metric_fn_dict.items():
            computed_metric = fn(y_pred_rescaled, y_true_rescaled)
            if computed_metric.size():
                record_metric_dict.setdefault(k, []).append(computed_metric)
            else:
                total_metric_dict[k] += computed_metric * data.num_graphs
 
    metric_dict = {k: total_metric_dict[k] / len(loader.dataset) for k in total_metric_dict}
    return total_loss / len(loader.dataset), metric_dict, record_metric_dict, out
 
 
def test_one_epoch(
    model,
    loader, 
    mask_rate, 
    device,
    mean, 
    std, 
    criterion, 
    metric_fn_dict,
):
    model.eval()
    total_loss = 0
    total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
    record_metric_dict = {}
 
    with torch.no_grad():
        for data in loader:
            data.x           = data.x.to(device)
            data.y           = data.y.to(device)
            data.edge_index  = data.edge_index.to(device)
            
 
            num_nodes  = torch.unique(data.batch, return_counts=True)[1]
            batch_mask = generate_batch_mask(num_nodes=num_nodes, mask_rate=mask_rate, required_idx=[])
            data.x[batch_mask] = 0
            x_input = torch.cat([data.x, data.pe.to(device)], dim=-1) if hasattr(data, "pe") else data.x
            eig     = data.eig.to(device) if hasattr(data, "eig") else None
            out     = model(x_input, data.edge_index, eig)
 
            y_pred = out[batch_mask]
            y_true = data.y[batch_mask]
            y_pred_rescaled = descale(scaled_data=y_pred, mean=mean, std=std)
            y_true_rescaled = descale(scaled_data=y_true, mean=mean, std=std)
 
            val_loss = criterion(y_pred, y_true)
            total_loss += float(val_loss) * data.num_graphs
            for k, fn in metric_fn_dict.items():
                computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                if computed_metric.size():
                    record_metric_dict.setdefault(k, []).append(computed_metric)
                else:
                    total_metric_dict[k] += computed_metric * data.num_graphs
 
    metric_dict = {k: total_metric_dict[k] / len(loader.dataset) for k in total_metric_dict}
    return total_loss / len(loader.dataset), metric_dict, record_metric_dict
 
 
def train(args, model, train_ds, val_ds, do_load):
    # edge_attrs   = args.use_data_edge_attrs.split(",") if args.use_data_edge_attrs is not None else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else args.device
 
    assert model is not None
    args.model_name = model.name if model.name is not None else type(model).__name__
 
    print(model)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
 
    if do_load:
        if not hasattr(args, "model_path"):
            print("model_path not found!")
        elif not os.path.exists(args.model_path):
            raise FileNotFoundError(f"{args.model_path} not found")
        else:
            model, cp_dict = load_checkpoint(args.model_path, model)
 
    model = model.to(device)
 
    print("#" * 80)
    print("args list:")
    for k, v in vars(args).items():
        print(f"  {k} = {v}")
    print("#" * 80)
 
    early_stop = EarlyStopping(mode="min", min_delta=args.min_delta, patience=args.patience)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if do_load and 'cp_dict' in dir() and cp_dict.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(cp_dict['optimizer_state_dict'])
        print(f"Resumed optimizer state from checkpoint (epoch {cp_dict.get('epoch', '?')})")

    
    criterion = torch.nn.MSELoss(reduction="mean").to(device)
    
    # Restore best_loss and start_epoch from checkpoint if resuming
    if do_load and 'cp_dict' in dir() and cp_dict is not None:
        best_loss = cp_dict.get('loss', np.inf)
        start_epoch = cp_dict.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}, best_loss so far: {best_loss:.6f}")
    else:
        best_loss = np.inf
        start_epoch = 1
    best_epoch = start_epoch - 1
    best_metric_dict = best_record_metric_dict = {}
    
    os.makedirs(args.save_path, exist_ok=True)
 
    train_metric_fn_dict = get_metric_fn_collection(prefix="train")
    val_metric_fn_dict   = get_metric_fn_collection(prefix="val")
 
    mean    = train_ds.mean
    std     = train_ds.std

 
 
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_metric_dict, tr_record_metric_dict, out = train_one_epoch(
            model=model, 
            optimizer=optimizer, 
            loader=train_loader,
            mask_rate=args.mask_rate, 
            device=device,
            mean=mean, 
            std=std, 
            criterion=criterion,
            metric_fn_dict=train_metric_fn_dict, 
        )
 
        val_loss, val_metric_dict, val_record_metric_dict = test_one_epoch(
            model=model, 
            loader=val_loader,
            mask_rate=args.mask_rate, 
            device=device,
            mean=mean, 
            std=std,
            criterion=criterion,
            metric_fn_dict=val_metric_fn_dict, 
        )
 
        if val_loss < best_loss:
            best_loss = val_loss
            best_metric_dict = val_metric_dict
            best_record_metric_dict = val_record_metric_dict
            best_epoch = epoch
            save_checkpoint(
                path=os.path.join(args.save_path, f"best_{args.model_name}.pth"),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict() if optimizer else None,
                epoch=best_epoch, 
                loss=best_loss,
                val_metric_dict=best_metric_dict,
                val_record_metric_dict=best_record_metric_dict,
                mean=train_ds.mean, 
                std=train_ds.std,
            )
 
        if epoch == 1 or (epoch % 5) == 0:
            print_metrics(epoch=epoch, tr_loss=tr_loss, val_loss=val_loss,
                          tr_metric_dict=tr_record_metric_dict, val_metric_dict=val_metric_dict)
            if not math.isnan(tr_loss):
                save_checkpoint(
                    path=os.path.join(args.save_path, f"last_{args.model_name}.pth"),
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict() if optimizer else None,
                    epoch=best_epoch, 
                    loss=best_loss,
                    val_metric_dict=val_metric_dict,
                    val_record_metric_dict=val_record_metric_dict,
                    mean=train_ds.mean, 
                    std=train_ds.std,
                    )

 
        if early_stop.step(torch.tensor(val_loss)):
            print(f"\n!! No improvement for {args.patience} epochs. Training stopped!")
            break
 
 
if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])
    set_seed(args.seed) 
    args, model = select_model(args, args.model_path != "")
    train_ds, val_ds = get_default_datasets(args)
    train(args=args, model=model, train_ds=train_ds, val_ds=val_ds, do_load=args.do_load)
 