
import sys
import os
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from utils.DataLoader import WDNDataset, get_stacked_set2
from utils.auxil import *
from models.ConfigModels import  select_model
import epynet
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Callable, Any
from collections import defaultdict


"""
Configurations for training and evaluation of GNNs on WDNs
"""
_MODEL             = "gatres_small"
_NUM_TEST_TRIALS    = 10
_MASK_RATE         = 0.95
_DATASET_PATHS = ["/content/datasets/ctown.zip"]
_INPUT_PATHS   = ["/content/inputs/ctown.inp"]
_MODEL_NAME        = None
_BATCH_SIZE    = 8
_DEVICE            = "cuda"
_SAVE_PATH     = "/content/drive/MyDrive/WDN/checkpoints/ctown_gatres"
_MODEL_PATH    = "/content/drive/MyDrive/WDN/checkpoints/ctown_gatres10k_12/best_GATResMeanConv_small_znorm_15b_32c_20260510_1654.pth"
_TEST_DATA_PATH = "/content/drive/MyDrive/WDN/datasets/ctown_24h_pattern_test.zip"
_TEST_INPUT_PATH = "/content/drive/MyDrive/WDN/inputs/ctown.inp"
_RWPE_STEPS         = 0
_FEATURE             = "pressure"
_RANDOM_SEED = 42

def get_arguments(raw_args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=_MODEL,
        type=str,
        choices=["gatres_small", "gatres_small_rwpe", "gatres_large", "gatres_small_signnet"],
        help="support model selection only.",
    )
    parser.add_argument("--model_path", default=_MODEL_PATH, type=str, help="Model path")
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

    parser.add_argument("--test_data_path", default=_TEST_DATA_PATH, type=str, help="timed dataset path for testing")  # 24hour
    parser.add_argument("--test_input_path", default=_TEST_INPUT_PATH, type=str, help="timed input path for testing")
    parser.add_argument("--feature", default=_FEATURE, choices=["pressure", "head"], type=str, help="feature input")
    parser.add_argument("--model_name", default=_MODEL_NAME, type=str, help="Name of model. Keep its empty to use the name of class by default")
    parser.add_argument("--batch_size", default=_BATCH_SIZE, type=int, help="batch size")
    parser.add_argument("--rwpe_steps", default=_RWPE_STEPS, type=int, help="Number of Random Walk PE steps appended to node features. 0 = disabled.")
    parser.add_argument("--lapev_k", default=0, type=int, help="Number of Laplacian eigenvectors for SignNet PE. 0 = disabled.")
    parser.add_argument(
        "--device",
        default=_DEVICE,
        type=str,
        choices=["cuda", "cpu"],
        help="Training device. If gpu is unavailable, device is set to cpu. Support: cuda| cpu",
    )
    
    parser.add_argument(
        "--num_test_trials",
        default=_NUM_TEST_TRIALS,
        type=int,
        help="Repeat the inference on test set N times with diff masks. The report will include mean and std in N times",
    )
    parser.add_argument("--seed", default=_RANDOM_SEED, type=int, help="Random seed for reproducibility")


    args = parser.parse_args(args=raw_args)
    return args


def get_default_datasets(args: argparse.Namespace, mean_dmd=0.1, std_dmd=1.0) -> tuple[WDNDataset, WDNDataset]:
    rwpe_steps = getattr(args, "rwpe_steps", 0)
    lapev_k    = getattr(args, "lapev_k", 0)

    train_ds = WDNDataset(
        zip_file_paths=args.dataset_paths,
        input_paths=args.input_paths,
        feature=args.feature,
        from_set="train",
        mean=None,
        std=None,
        norm_type="znorm",
        rwpe_steps=rwpe_steps,
        lapev_k=lapev_k,
    )

    test_ds = get_stacked_set2(
        zip_file_path=args.test_data_path,  # fullnode
        input_path=args.test_input_path,
        feature=args.feature,
        train_mean=train_ds.mean,
        train_std=train_ds.std,
        rwpe_steps=rwpe_steps,
        lapev_k=lapev_k,
    )
    return train_ds, test_ds


def test_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    mask_rate: float,
    device: str,
    mean: Any,
    std: Any,
    criterion: Callable,
    metric_fn_dict: dict,
) -> tuple[float, dict]:
    
    model.eval()

    total_loss = 0
    total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
    created_all_mask = False
    all_mask = None

    with torch.no_grad():
        for data in loader:
            data.x = data.x
            data.y = data.y.to(device)
            data.edge_index = data.edge_index.to(device)

            num_nodes = torch.unique(data.batch, return_counts=True)[1]

            data_x1 = torch.clone(data.x).to(device)

            
            all_mask = generate_batch_mask(num_nodes=num_nodes, mask_rate=mask_rate, required_idx=[])

            data_x1[all_mask] = 0

            x_input = torch.cat([data_x1, data.pe.to(device)], dim=-1) if hasattr(data, "pe") else data_x1
            eig     = data.eig.to(device) if hasattr(data, "eig") else None
            out     = model(x_input, data.edge_index, eig)

            y_pred = out[all_mask]  # y_pred.masked_select(mask)
            y_true = data.y[all_mask]  # y_true.masked_select(mask)

            y_pred_rescaled = descale(scaled_data=y_pred, mean=mean, std=std)
            y_true_rescaled = descale(scaled_data=y_true, mean=mean, std=std)

            val_loss = criterion(y_pred, y_true)
            total_loss += float(val_loss) * data.num_graphs
            for k, fn in metric_fn_dict.items():
                computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                total_metric_dict[k] += computed_metric * data.num_graphs

    len_dataset = len(loader.dataset)
    test_loss = total_loss / len_dataset

    test_metric_dict = {k: total_metric_dict[k] / len_dataset for k in total_metric_dict.keys()}    

    test_metric_dict = {k: v for k, v in test_metric_dict.items()}

    return test_loss, test_metric_dict


def test_clean(
    model: torch.nn.Module,
    test_ds: WDNDataset,
    args: argparse.Namespace,
    device: str,
    mean: Any,
    std: Any,
    criterion: Callable,
    metric_fn_dict: dict,
) -> tuple[list, dict, list, dict]:
    test_losses = []
    test_metrics_dict = defaultdict(list)

    

    test_batch_size = args.batch_size
    repeat_test_time = args.num_test_trials
    loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    for i in tqdm(range(repeat_test_time)):
        trial = i
        test_loss, test_metric_dict = test_and_collect_once(
            model=model,
            loader=loader,
            trial=trial,
            args=args,
            device=device,
            mean=mean,
            std=std,
            criterion=criterion,
            metric_fn_dict=metric_fn_dict,
        )

        test_losses.append(test_loss)
        for k in test_metric_dict.keys():
            test_metrics_dict[k].append(test_metric_dict[k])

    return test_losses, test_metrics_dict

def test_and_collect_once(
    model: torch.nn.Module,
    loader: DataLoader,
    trial: int,
    args: argparse.Namespace,
    device: str,
    mean: Any,
    std: Any,
    criterion: Callable,
    metric_fn_dict: dict,
) -> tuple[float, dict, float, dict]:
    # for all nodes
    test_loss, test_metric_dict = test_one_epoch(
        model=model,
        loader=loader,
        mask_rate=args.mask_rate,
        device=device,
        mean=mean,
        std=std,
        criterion=criterion,
        metric_fn_dict=metric_fn_dict,
    )

    return test_loss, test_metric_dict


def test(
    args: argparse.Namespace, model: torch.nn.Module, train_ds: WDNDataset, test_ds: WDNDataset, do_load: bool = True
) -> tuple[dict, dict, dict]:
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else args.device

    assert model is not None
    args.model_name = model.name if model.name is not None else type(model).__name__

    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))
    
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
        print(f"{k} = {v}")
    print("#" * 80)

    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))

    criterion = torch.nn.MSELoss(reduction="mean").to(device)

    test_metric_fn_dict = get_metric_fn_collection(prefix="test")
    mean = train_ds.mean
    std = train_ds.std


    test_losses, test_metrics_dict = test_clean(
        model=model,
        test_ds=test_ds,
        args=args,
        device=device,
        mean=mean,
        std=std,
        criterion=criterion,
        metric_fn_dict=test_metric_fn_dict,
    )
    

    trials = len(test_losses)

    mean_test_loss, std_test_loss = np.mean(test_losses), np.std(test_losses) + 1e-6

    out_test_metric_dict = {}
    for k in test_metrics_dict.keys():
        x = torch.tensor(test_metrics_dict[k])
        out_test_metric_dict[f"{k}_mean"], out_test_metric_dict[f"{k}_std"] = torch.mean(x), torch.std(x) + 1e-6

    print_multitest_metrics(
        trials=trials,
        mean_test_loss=mean_test_loss,
        std_test_loss=std_test_loss,
        out_test_metric_dict=out_test_metric_dict,
    )

    out_test_loss_dict = {
        "test_loss_mean": mean_test_loss,
        "test_loss_std": std_test_loss,
    }

    return out_test_loss_dict, out_test_metric_dict





if __name__ == "__main__":

    args = get_arguments(sys.argv[1:])
    set_seed(args.seed)
    args, model = select_model(args, args.model_path != "")
    train_ds, test_ds = get_default_datasets(args)

    test(args=args, model=model, train_ds=train_ds, test_ds=test_ds, do_load=True)

