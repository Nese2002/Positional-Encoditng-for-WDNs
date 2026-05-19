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

    def forward(self, x, edge_index, eig=None):  # eig unused; accepted for uniform call signature
        x = self.lin0(x)
        for block in self.blocks:
            x = block(x, edge_index)
        return self.lin1(x)



class SignNetPE(torch.nn.Module):


    def __init__(self, k: int, phi_hidden: int = 64, pe_dim: int = 16):
        super().__init__()
        self.phi = torch.nn.Sequential(
            Linear(1, phi_hidden),
            torch.nn.ReLU(),
            Linear(phi_hidden, phi_hidden),
        )
        self.rho = torch.nn.Sequential(
            Linear(phi_hidden, pe_dim),
            torch.nn.ReLU(),
        )

    def forward(self, eig: torch.Tensor) -> torch.Tensor:
        # eig: [N, k]
        h = eig.unsqueeze(-1)            # [N, k, 1]
        h = self.phi(h) + self.phi(-h)   # [N, k, phi_hidden]  — sign-invariant
        h = h.sum(dim=1)                 # [N, phi_hidden]      — aggregate over k
        return self.rho(h)               # [N, pe_dim]


class GATResMeanConvSignNet(torch.nn.Module):
    def __init__(
        self,
        name: str = "GATResMeanConvSignNet",
        num_blocks: int = 15,
        nc: int = 32,
        k: int = 16,
        phi_hidden: int = 64,
        pe_dim: int = 16,
    ):
        super().__init__()
        self.name = name
        self.num_blocks = num_blocks
        self.sign_net = SignNetPE(k=k, phi_hidden=phi_hidden, pe_dim=pe_dim)
        self.lin0 = Linear(1 + pe_dim, nc)
        self.blocks = torch.nn.ModuleList(
            GResBlockMeanConv(nc, nc, nc) for _ in range(num_blocks)
        )
        self.lin1 = Linear(nc, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, eig: torch.Tensor = None) -> torch.Tensor:
        assert eig is not None, "GATResMeanConvSignNet requires eigenvectors passed as eig (data.eig)"
        pe = self.sign_net(eig)              # [N, pe_dim]
        x = torch.cat([x, pe], dim=-1)      # [N, 1 + pe_dim]
        x = self.lin0(x)
        for block in self.blocks:
            x = block(x, edge_index)
        return self.lin1(x)