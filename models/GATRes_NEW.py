import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SimpleConv
from torch.nn import Linear


class GResBlockMeanConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hc):
        super().__init__()
        self.conv1 = GATConv(in_dim, hc, 2, concat=True)
        self.conv2 = GATConv(hc * 2, out_dim, 1, concat=False)
        self.mean_conv = SimpleConv(aggr="mean")

    def forward(self, x, edge_index):
        x_0 = torch.clone(x)
        x = self.conv1(x, edge_index,).relu()
        x = self.conv2(x, edge_index)
        x = self.mean_conv(x, edge_index) + x_0
        return F.relu(x)


class GATResMeanConv(torch.nn.Module):
    def __init__(self, name="GATResMeanConv", num_blocks=15, nc=32, in_dim=1):
        super().__init__()
        self.name = name
        self.num_blocks = num_blocks
        self.lin0 = Linear(in_dim, nc)
        self.blocks = torch.nn.ModuleList(
            GResBlockMeanConv(nc, nc, nc) for _ in range(num_blocks)
        )
        self.lin1 = Linear(nc, 1)

    def forward(self, x, edge_index):
        x = self.lin0(x)
        for block in self.blocks:
            x = block(x, edge_index)
        return self.lin1(x)