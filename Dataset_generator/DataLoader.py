import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
import zarr
import torch_geometric.utils as pgu
from torch_geometric.data import Dataset, Data
from wntr.network import WaterNetworkModel

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GNN_SOURCE_DIR = _PROJECT_ROOT / "gnn-pressure-estimation" / "gnn_pressure_estimation"
if str(_GNN_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_GNN_SOURCE_DIR))

from utils.auxil import scale, nx_to_pyg


def compute_rwpe(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    from torch_geometric.utils import degree
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes).clamp(min=1)
    A = torch.zeros(num_nodes, num_nodes)
    A[row, col] = 1.0
    RW = (1.0 / deg).unsqueeze(1) * A
    pe, RW_k = [], RW.clone()
    for _ in range(k):
        pe.append(RW_k.diagonal().clone())
        RW_k = RW_k @ RW
    return torch.stack(pe, dim=1)  # [N, k]


def _build_graph_template(graph: nx.Graph, edge_attrs: bool = False, rwpe_steps: int = 0) -> Data:
    data = pgu.from_networkx(graph)
    if edge_attrs:
        data.edge_attr = deepcopy(data.weight)
        for attr in ("weight", "type"):
            if hasattr(data, attr):
                delattr(data, attr)
    for attr in ("pos", "edge_type"):
        if hasattr(data, attr):
            delattr(data, attr)
    if rwpe_steps > 0:
        data.pe = compute_rwpe(data.edge_index, data.num_nodes, rwpe_steps)
    return data


def _resolve_keep_list(
    wn: WaterNetworkModel,
    removal: str,
    root: Optional[zarr.Group],
    feature: str,
) -> Optional[list[str]]:
    if removal == "keep_list":
        if root and "ordered_name_list" in root.attrs:
            return root.attrs["ordered_name_list"]
        if root and "ordered_names_by_attr" in root.attrs and feature in root.attrs["ordered_names_by_attr"]:
            return root.attrs["ordered_names_by_attr"][feature]
        print("WARN! ordered_name_list/ordered_names_by_attr not found in zarr; falling back to keep_junction.")
        return wn.junction_name_list
    if removal == "reservoir":
        return list(set(wn.node_name_list).difference(wn.reservoir_name_list)) if wn.reservoir_name_list else None
    if removal == "tank":
        return list(set(wn.node_name_list).difference(wn.tank_name_list)) if wn.tank_name_list else None
    if removal == "keep_junction":
        return wn.junction_name_list
    return None  # keep_all


class WDNDataset(Dataset):
    """PyTorch Geometric dataset for WDN simulation snapshots stored in zarr zip archives.

    Supports multiple networks, lazy or eager conversion to PyG Data objects,
    and optional z-score / min-max normalisation.
    """

    _VALID_REMOVALS = frozenset({"keep_list", "reservoir", "tank", "keep_junction", "keep_all"})
    _VALID_EDGE_ATTRS = frozenset({"diameter", "length", "valve_mask"})

    def __init__(
        self,
        input_paths: list[str],
        zip_file_paths: list[str],
        feature: str,
        from_set: str,
        num_records: Optional[int] = None,
        removal: str = "keep_list",
        do_scale: bool = True,
        mean=None,
        std=None,
        min=None,
        max=None,
        lazy_convert_pygdata: bool = False,
        edge_attrs: Optional[list[str]] = None,
        edge_mean=None,
        edge_std=None,
        edge_min=None,
        edge_max=None,
        norm_type: str = "znorm",
        rwpe_steps: int = 0,
        **kwargs,
    ):
        assert norm_type in ("znorm", "minmax", "unused")
        assert removal in self._VALID_REMOVALS, f"removal must be one of {self._VALID_REMOVALS}"
        assert edge_attrs is None or set(edge_attrs).issubset(self._VALID_EDGE_ATTRS)
        assert len(input_paths) == len(zip_file_paths)

        self._roots: list = []
        self._templates: list[Data] = []
        self._lengths: list[int] = []
        self._keeplists: list[list[str]] = []
        raw_arrays: list[np.ndarray] = []

        for inp, zfp in zip(input_paths, zip_file_paths):
            assert os.path.isfile(inp) and inp.endswith((".inp", ".net")), f"{inp} is not an .inp/.net file"
            template, array, keep_list = self._collect(
                inp, zfp, feature, edge_attrs, removal, from_set, num_records,
                rwpe_steps=rwpe_steps, **kwargs,
            )
            self._templates.append(template)
            self._lengths.append(array.shape[0])
            raw_arrays.append(array)
            self._keeplists.append(keep_list)

        self.cumsum_lengths = np.cumsum(self._lengths)
        self.feature = feature
        self.from_set = from_set
        self.length = sum(self._lengths)
        self.transform = None
        self.lazy_convert_pygdata = lazy_convert_pygdata
        self.norm_type = norm_type
        self.num_arrays = len(raw_arrays)

        flat = np.concatenate([a.flatten() for a in raw_arrays])
        self.mean = float(np.mean(flat)) if mean is None else mean
        self.std  = float(np.std(flat))  if std  is None else std
        self.min  = float(np.min(flat))  if min  is None else min
        self.max  = float(np.max(flat))  if max  is None else max
        self.edge_mean = self.edge_std = self.edge_min = self.edge_max = None

        if do_scale and norm_type != "unused":
            scale_kw = dict(norm_type=norm_type, mean=self.mean, std=self.std, min=self.min, max=self.max)
            raw_arrays = [scale(a, **scale_kw) for a in raw_arrays]

            if edge_attrs:
                flat_e = np.concatenate([t.edge_attr for t in self._templates], axis=0)
                self.edge_mean = np.mean(flat_e, axis=0) if edge_mean is None else edge_mean
                self.edge_std  = np.std(flat_e,  axis=0) if edge_std  is None else edge_std
                self.edge_min  = np.min(flat_e,  axis=0) if edge_min  is None else edge_min
                self.edge_max  = np.max(flat_e,  axis=0) if edge_max  is None else edge_max
                edge_scale_kw = dict(
                    norm_type=norm_type,
                    mean=self.edge_mean, std=self.edge_std,
                    min=self.edge_min,   max=self.edge_max,
                )
                self._templates = [
                    self._apply_edge_scale(t, **edge_scale_kw) for t in self._templates
                ]

        self._indices = range(self.length)

        if lazy_convert_pygdata:
            self._arrays = raw_arrays
        else:
            self._arrays = [
                nx_to_pyg(data=raw_arrays[arr_id][i], graph=self._templates[arr_id])
                for arr_id in range(len(raw_arrays))
                for i in range(len(raw_arrays[arr_id]))
            ]

    # ------------------------------------------------------------------
    # PyG Dataset interface
    # ------------------------------------------------------------------

    def len(self) -> int:
        return self.length

    def get(self, idx: int) -> Data:
        if not self.lazy_convert_pygdata:
            return self._arrays[idx]
        arr_id = next(i for i, cl in enumerate(self.cumsum_lengths) if idx < cl)
        offset = self.cumsum_lengths[arr_id - 1] if arr_id > 0 else 0
        return nx_to_pyg(data=self._arrays[arr_id][idx - offset], graph=self._templates[arr_id])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect(
        self,
        input_path: str,
        zip_file_path: str,
        feature: str,
        edge_attrs: Optional[list[str]],
        removal: str,
        from_set: str,
        num_records: Optional[int],
        **kwargs,
    ) -> tuple[Data, np.ndarray, list[str]]:
        rwpe_steps = kwargs.pop("rwpe_steps", 0)
        assert from_set in ("train", "valid", "test"), f"from_set '{from_set}' not supported"
        assert os.path.isfile(zip_file_path) and zip_file_path.endswith(".zip"), \
            f"{zip_file_path} is not a .zip file"

        root = zarr.open(store=zip_file_path, mode="r")
        assert feature in root.group_keys(), f"feature '{feature}' not found in {zip_file_path}"
        self._roots.append(root)

        wn = WaterNetworkModel(input_path)
        link_weight_dict = self._build_link_weights(wn, edge_attrs)
        graph = nx.Graph(wn.to_graph(link_weight=link_weight_dict)).to_undirected()
        keep_list = _resolve_keep_list(wn=wn, removal=removal, root=root, feature=feature)

        array = np.array(root[feature][from_set])
        if num_records is not None:
            array = array[:num_records]

        if keep_list is not None:
            indices = [i for i, name in enumerate(wn.node_name_list) if name in keep_list]
            array = np.take(array, indices, axis=-1)
            assert array.shape[-1] >= len(keep_list)

        subgraph = graph.subgraph(keep_list).copy() if keep_list is not None else graph
        template = _build_graph_template(subgraph, edge_attrs=bool(edge_attrs), rwpe_steps=rwpe_steps)
        return template, array, keep_list or wn.node_name_list

    @staticmethod
    def _build_link_weights(wn: WaterNetworkModel, edge_attrs: Optional[list[str]]) -> Optional[dict]:
        if not edge_attrs:
            return None
        weights = pd.concat([wn.query_link_attribute(attribute=a) for a in edge_attrs], axis=1).fillna(0)
        weight_dict = weights.T.to_dict(orient="list")
        zeros = np.zeros(len(next(iter(weight_dict.values()))), dtype=weights.dtypes[0])
        for uid in wn.link_name_list:
            if uid not in weight_dict:
                weight_dict[uid] = zeros.copy()
        return weight_dict

    @staticmethod
    def _apply_edge_scale(template: Data, **scale_kwargs) -> Data:
        template.edge_attr = scale(template.edge_attr, **scale_kwargs)
        return template
