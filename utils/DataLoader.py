import os
from torch_geometric.data import Dataset, Data
import zarr
from wntr.network import WaterNetworkModel
from wntr.sim.epanet import EpanetSimulator
from utils.auxil import scale, nx_to_pyg
import epynet
import torch_geometric.utils as pgu
import networkx as nx
import numpy as np
from copy import deepcopy
import torch

from typing import Any, Optional


def compute_rwpe(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    from torch_geometric.utils import degree
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes).clamp(min=1)
    A = torch.zeros(num_nodes, num_nodes)
    A[row, col] = 1.0
    RW = (1.0 / deg).unsqueeze(1) * A          # D⁻¹A  [N, N]
    pe, RW_k = [], RW.clone()
    for _ in range(k):
        pe.append(RW_k.diagonal().clone())      # self-return probability at step i
        RW_k = RW_k @ RW
    return torch.stack(pe, dim=1)               # [N, k]


def compute_lapev(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    """Compute the k smallest non-trivial eigenvectors of the symmetric
    normalised Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.

    Uses dense eigendecomposition for graphs with <= 2000 nodes and the
    sparse Lanczos solver (eigsh) for larger ones.  Trivial eigenvectors
    (eigenvalue < 1e-5, corresponding to connected components) are
    discarded.  If fewer than k non-trivial eigenvectors exist the output
    is zero-padded to shape [N, k].

    Returns:
        Tensor of shape [N, k], dtype float32.
    """
    from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
    import numpy as np

    lap_ei, lap_ew = get_laplacian(edge_index, num_nodes=num_nodes, normalization="sym")
    L = to_scipy_sparse_matrix(lap_ei, lap_ew, num_nodes)

    if num_nodes <= 2000:
        eigenvalues, eigenvectors = np.linalg.eigh(L.toarray().astype(np.float64))
    else:
        import scipy.sparse.linalg as sla
        k_req = min(k + 2, num_nodes - 2)
        eigenvalues, eigenvectors = sla.eigsh(L.astype(np.float64), k=k_req, which="SM")
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    # Drop near-zero eigenvalues (trivial, one per connected component)
    nontrivial = eigenvalues > 1e-5
    eigenvectors = eigenvectors[:, nontrivial][:, :k]

    if eigenvectors.shape[1] < k:
        pad = np.zeros((num_nodes, k - eigenvectors.shape[1]))
        eigenvectors = np.concatenate([eigenvectors, pad], axis=1)

    return torch.from_numpy(eigenvectors.astype(np.float32))  # [N, k]


def get_graph_template(new_graph: nx.Graph, rwpe_steps: int = 0, lapev_k: int = 0) -> Data:
    graph_template = pgu.from_networkx(new_graph)
    del graph_template.pos
    del graph_template.edge_type

    if rwpe_steps > 0:
        graph_template.pe = compute_rwpe(graph_template.edge_index, graph_template.num_nodes, rwpe_steps)

    if lapev_k > 0:
        graph_template.eig = compute_lapev(graph_template.edge_index, graph_template.num_nodes, lapev_k)

    return graph_template

class WDNDataset(Dataset):
    def __init__(
        self,
        input_paths,
        zip_file_paths,
        feature,
        from_set,
        mean=None,
        std=None,
        norm_type="znorm",
        rwpe_steps=0,
        lapev_k=0,
        **kwargs,
    ):
        """The dataset class supports multiple datasets

        :param List<str> input_paths: list of inp files
        :param List<str> zip_file_paths: list of zip files
        :param str feature: supported features: pressure/ head/ flow
        :param str from_set: supported sets: train/ valid/ test
        :param str removal: supported removals: keep_list/ reservoir/ tank/ keep_junction, defaults to 'keep_list'
        :param mean: existing computed mean, set None to re-compute, defaults to None
        :param std: existing computed std, set None to re-compute, defaults to None
        :param norm_type: normalization type supports: znorm/minmax/unused, default is znorm
        :param int rwpe_steps: number of Random Walk PE steps. 0 = disabled, defaults to 0
        :param int lapev_k: number of Laplacian eigenvectors for SignNet PE. 0 = disabled, defaults to 0
        :raises KeyError: Key is not found in attrs of the zarr file
        """
        assert norm_type == "znorm" 
        assert len(input_paths) == len(zip_file_paths)
        self._roots = []
        self._templates = []
        self._lengths = []
        self.template_dict = {}

        # print(f'{self.__class__.__name__}-removal = {removal}')
        # assert removal == "keep_junction", (
        #     f"Removal only supports keep_list,reservoir,tank,keep_junction. Got {removal}"
        # )
        _arrays = []
        _keeplists = []
        for i, (input_path, zip_file_path) in enumerate(zip(input_paths, zip_file_paths)):
            assert os.path.isfile(input_path) and (input_path[-4:] == ".inp" or input_path[-4:] == ".net"), f"{input_path} is not a INP/ NET file"

            graph_template, array, keep_list = self.collect(input_path, zip_file_path, feature, from_set, rwpe_steps=rwpe_steps, lapev_k=lapev_k, **kwargs)  #For each network it calls self.collect() (explained below) which returns three things: a graph template (the fixed topology), a numpy array of simulation results, and a keep list (which nodes to retain). All three are stored in parallel lists (_templates, _arrays, _keeplists).

            self._templates.append(graph_template)
            self._lengths.append(array.shape[0])
            _arrays.append(array)
            _keeplists.append(keep_list)

        # After processing all input files, the constructor computes cumulative lengths to facilitate indexing across multiple datasets.
        self._keeplists = _keeplists
        self._arrays = _arrays
        self._ids = np.cumsum(self._lengths[:-1])
        self.cumsum_lengths = np.cumsum(self._lengths)
        self.feature = feature
        self.from_set = from_set
        self.length = sum(self._lengths)
        self.transform = None
        self.norm_type = norm_type

        flatten_arr = np.concatenate([arr.flatten() for arr in self._arrays])
        
        # Compute mean, std for normalization. If any of these are provided (not None), use the provided value instead of computing from the data
        self.mean = np.mean(flatten_arr) if mean is None else mean
        self.std = np.std(flatten_arr) if std is None else std
        
        for i in range(len(self._arrays)):
            self._arrays[i] = scale(self._arrays[i], mean=self.mean, std=self.std)


        self._indices = range(self.length)
        self.num_arrays = len(self._arrays)

        # eager mode: every simulation scenario is immediately converted from numpy to a torch_geometric.data.Data object at init time. This uses more memory upfront but makes get() faster. If True, conversion happens on-demand in get().
        tmp_array = []
        for arr_id in range(len(self._arrays)):
            for internal_id in range(len(self._arrays[arr_id])):
                tmp_array.append(nx_to_pyg(data=self._arrays[arr_id][internal_id], graph=self._templates[arr_id]))

        self._arrays = tmp_array

    def size(self) -> int:
        return len(self._arrays)

    def len(self) -> int:
        return self.length

    # The get method retrieves the data for a given index. It simply returns the pre-converted Data object from _arrays.
    def get(self, idx):
        return self._arrays[idx]

    def collect(
        self,
        input_path: str,
        zip_file_path: str,
        feature: str,
        from_set: str,
        rwpe_steps: int = 0,
        lapev_k: int = 0,
        **kwargs
    ) -> tuple[Data, np.ndarray, list[str]]:

        assert os.path.isfile(zip_file_path) and zip_file_path[-4:] == ".zip", f"{zip_file_path} is not a zip file"
        assert from_set in ["train", "valid", "test"], f"from_set {from_set} is not supported"

        # Open the zarr file and check if the required feature is available. The zarr file is expected to contain the simulation results for the specified feature (e.g., pressure, head, flow) for the given input network. The root variable will be used to access the data for the specified feature and from_set (train/ valid/ test).
        root = zarr.open(store=zip_file_path, mode="r")
        assert set([feature]).issubset(root.group_keys()), f"feature {feature} is unavailabel in zarr file {zip_file_path}"

        self._roots.append(root)
        wn = WaterNetworkModel(input_path)

        # The code creates an undirected graph representation of the water network using the wn.to_graph() method. If edge attributes were specified and processed into link_weight_dict, those weights are included in the graph construction. The resulting graph is then converted to an undirected graph using to_undirected(). This graph will be used as the template for the PyG Data objects that will be created for each simulation scenario.
        graph = nx.Graph(wn.to_graph(link_weight=None)).to_undirected()

        # Depending on the specified removal strategy, the code determines which nodes to keep in the graph. The get_keep_list function checks the removal parameter and constructs a list of node names to retain based on the water network model and the contents of the zarr file. For example, if removal is "keep_junction", it will return a list of junction node names. If removal is "reservoir", it will return all nodes except reservoirs. This keep_list is then used to create a subgraph that contains only the desired nodes, which will be the basis for the graph template and the corresponding feature array.
        keep_list = wn.junction_name_list

        array = np.array(root[feature][from_set])

        taken_indices = []
        for i, name in enumerate(wn.node_name_list):
            if name in keep_list:
                taken_indices.append(i)
        
        # The code takes the original array of feature values (e.g., pressure) and selects only the columns corresponding to the nodes in the keep_list. This is done by calculating the indices of the nodes to keep and using np.take() to select those columns from the array. This ensures that the feature array aligns with the subgraph defined by the keep_list, which will be used to create the PyG Data objects.
        array = np.take(array, taken_indices, axis=-1)

        assert array.shape[-1] >= len(keep_list)

        new_graph = graph.subgraph(keep_list).copy() if keep_list is not None else graph

        graph_template = get_graph_template(new_graph=new_graph, rwpe_steps=rwpe_steps, lapev_k=lapev_k)

        return graph_template, array, keep_list 



def get_stacked_set2(
    zip_file_path: str,
    input_path: str,
    train_mean: Any,
    train_std: Any,
    feature: str = "pressure",
    rwpe_steps: int = 0,
    lapev_k: int = 0,
    from_set: str = "test",
):
    current_records = 0
    test_train_ds = WDNDataset(
        zip_file_paths=[zip_file_path],
        input_paths=[input_path],
        feature=feature,
        from_set=from_set,
        mean=train_mean,
        std=train_std,
        rwpe_steps=rwpe_steps,
        lapev_k=lapev_k,
    )
    current_records += len(test_train_ds)
    ret_test_ds = test_train_ds

    
    return ret_test_ds