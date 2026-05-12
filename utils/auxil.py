import numpy as np
from copy import deepcopy
import torch
from functools import partial
import torch.nn.functional as F
from typing import Union, Any
import wandb
import networkx as nx

def calculate_nse(y_pred, y_true, exponent=2):
    raveled_y_pred = torch.ravel(y_pred)
    raveled_y_true = torch.ravel(y_true)
    return 1.0 - torch.div(
        torch.sum(torch.pow(raveled_y_pred - raveled_y_true, exponent)),
        torch.sum(torch.pow(raveled_y_true - torch.mean(raveled_y_true), exponent)) + 1e-12,
    )


def calculate_rmse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def calculate_rel_error(y_pred, y_true):
    err = torch.abs(torch.subtract(y_true, y_pred))
    mask = torch.abs(y_true) > 0.01
    rel_err = torch.abs(torch.divide(err[mask], y_true[mask]))
    return torch.mean(rel_err)


def calculate_accuracy(y_pred, y_true, threshold=0.2):
    mae = torch.abs(torch.subtract(y_true, y_pred))
    acc = (mae <= (y_true * threshold)).float()
    return torch.mean(acc)


def calculate_correlation_coefficient(y_pred, y_true):
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

    return torch.clamp(cost, -1.0, 1.0)


def calculate_r2(y_pred, y_true):
    r = calculate_correlation_coefficient(y_pred, y_true)
    return r**2

def get_metric_fn_collection(prefix: str) -> dict:
    """
    util to create metric funtions

    Args:
        prefix (str): set a prefix name for tracking these experiment

    Returns:
        dict: contains functional name and callable functions
    """
    metric_fn_dict = {
        f"{prefix}_error": calculate_rel_error,
        f"{prefix}_0.1": partial(calculate_accuracy, threshold=0.1),
        f"{prefix}_corr": calculate_correlation_coefficient,
        f"{prefix}_r2": calculate_r2,
        f"{prefix}_mae": F.l1_loss,
        f"{prefix}_rmse": calculate_rmse,
        f"{prefix}_mynse": partial(calculate_nse, exponent=2),
    }
    return metric_fn_dict

def save_checkpoint(path: str, **kwargs) -> str:
    """support save checkpoint. User can leverage kwargs to store model and relevant data

    Args:
        path (str): saved path

    Returns:
        str: saved path
    """
    torch.save(kwargs, path)
    return path


def load_checkpoint(path: str, model: torch.nn.Module) -> tuple[torch.nn.Module, dict]:
    """Load model and relevant data

    Args:
        path (str): checkpoint file
        model (torch.nn.Module): model architecture to load weights into

    Returns:
        tuple[torch.nn.Module, dict]: tuple of loaded model and relevant data as dict
    """
    assert path[-4:] == ".pth"
    assert model is not None
    cp_dict = torch.load(path, weights_only=False)
    model.load_state_dict(cp_dict["model_state_dict"])
    return model, cp_dict


def mask_nodes(num_nodes: int, masking_rate: float, required_idx: list[int]) -> np.ndarray:
    """function supports to build a mask array

    Args:
        num_nodes (int): number of available nodes
        masking_rate (float): masking ratio
        required_idx (list[int]): indices required to be masked. Set none if unused

    Returns:
        np.ndarray: binary mask array
    """
    mask_length = int(num_nodes * masking_rate) - len(required_idx)
    assert mask_length > 0
    selected_nodes = list(set(range(num_nodes)).difference(required_idx))
    idx = np.random.choice(selected_nodes, mask_length, replace=False)
    mask = np.zeros(num_nodes)
    mask[idx] = 1
    mask[required_idx] = 1
    assert len(mask[mask == 1]) == int(num_nodes * masking_rate)
    mask = mask.astype(bool)
    return mask  # .reshape(-1, 1)




def generate_batch_mask(num_nodes: int, mask_rate: float, required_idx: list[int]) -> np.ndarray:
    """generate a batch of mask arrays

    Args:
       num_nodes (int): number of available nodes
        masking_rate (float): masking ratio
        required_idx (list[int]): indices required to be masked. Set none if unused

    Returns:
        np.ndarray: _description_
    """

    def decorator(i):
        return mask_nodes(i, mask_rate, required_idx)

    test = np.hstack(list(map(decorator, num_nodes)))
    return test

def scale(data: Any, mean: Any = None, std: Any = None, eps: float = 1e-8) -> Any:
    assert mean and std, "mean and std values are missing"
    return (data - mean) / (std + eps)

def descale(scaled_data: Any, mean: Any = None, std: Any = None) -> Any:
    """
    Descale function supports denormalization
    """
    assert mean and std, "mean and std values are missing"
    data = (scaled_data * std) + mean
    return data

def print_metrics(epoch: int, tr_loss: float, val_loss: float, tr_metric_dict: dict, val_metric_dict: dict):
    """
    support beautifying string format
    """
    metric_log = ""

    for k, v in tr_metric_dict.items():
        metric_log += f"{k}: {v:.4f}, "

    for k, v in val_metric_dict.items():
        metric_log += f"{k}: {v:.4f}, "

    print(f"Epoch: {epoch:03d}, train loss: {tr_loss:.4f}, val_loss: {val_loss:.4f}, {metric_log}")


def nx_to_pyg(data: Any, graph: "nx.graph") -> "torch_geometric.data.Data":
    """convert nx graph and data into pyg Data format

    Args:
        data (Any):
        graph (nx.graph): nx graph containing topology only

    Returns:
        torch_geometric.data.Data: pyg data format
    """
    g_data = deepcopy(graph)
    y = data
    g_data.y = torch.Tensor(np.reshape(y, [-1, 1]))
    g_data.x = torch.Tensor(np.reshape(y, [-1, 1]))
    return g_data
