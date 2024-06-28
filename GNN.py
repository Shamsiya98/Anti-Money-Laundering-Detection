import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

# Define GNN Model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, int(hidden_channels / 4), heads=1, concat=False, dropout=0.6)
        self.lin = nn.Linear(int(hidden_channels / 4), out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        embeddings = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.lin(embeddings)
        x = self.sigmoid(x)
        return x, embeddings