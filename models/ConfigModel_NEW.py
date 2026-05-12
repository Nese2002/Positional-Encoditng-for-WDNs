import argparse
import torch
from GATRes import GATResMeanConv


# def _apply_common_args(args: argparse.Namespace) -> argparse.Namespace:
#     args.criterion = "mse"
#     args.use_data_edge_attrs = None
#     args.norm_type = "znorm"
#     return args


def config_gatres_small(args: argparse.Namespace) -> tuple[argparse.Namespace, torch.nn.Module]:
    # args = _apply_common_args(args)
    # args.model_path = r"experiments_logs\simple_test\gatres_znorm\best_GATResMeanConv_znorm_20235922.pth"
    return args, GATResMeanConv(
        "GATRes_small",
        num_blocks=15, 
        nc=32,
    )


def config_gatres_large(args: argparse.Namespace) -> tuple[argparse.Namespace, torch.nn.Module]:
    return args, GATResMeanConv(
        "GATRes_large",
        num_blocks=25, 
        nc=128,
    )


def config_gatres_small_rwpe(args: argparse.Namespace) -> tuple[argparse.Namespace, torch.nn.Module]:
    k = getattr(args, "rwpe_steps", 16)  # read from args so --rwpe_steps controls it
    args.rwpe_steps = k
    return args, GATResMeanConv(
        f"GATRes_small_rwpe{k}",
        num_blocks=15, 
        nc=32, 
        in_dim=1 + k,
    )


# Registry: uniform (args, name) -> (args, model) signature for every entry
MODEL_REGISTRY: dict[str, callable] = {
    "gatres_small":      config_gatres_small,
    "gatres_large":      config_gatres_large,
    "gatres_small_rwpe": config_gatres_small_rwpe,
}

CHOICES = list(MODEL_REGISTRY.keys())


def select_model(
    args: argparse.Namespace,
    reset_model_path: bool = False,
) -> tuple[argparse.Namespace, torch.nn.Module]:
    model = getattr(args, "model", "gatres_small")
    old_model_path = getattr(args, "model_path", "")

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Valid choices: {CHOICES}")

    args, model = MODEL_REGISTRY[model](args)

    if reset_model_path:
        args.model_path = old_model_path

    return args, model