import torch
from torch_geometric.nn import GATv2Conv, GCNConv
from torch.nn import Linear, LeakyReLU, Dropout


class FirstEncoder(torch.nn.Module):
  def __init__(self, input_size, features_size, out_channels=8, in_head=8, dropout=0.5, kind='gcn'):
    super(FirstEncoder, self).__init__()
    self.kind = kind
    if self.kind == 'gat':
      self.conv1 = GATv2Conv(in_channels=input_size, out_channels=out_channels, heads=in_head, dropout=dropout)
      self.conv2 = GATv2Conv(in_channels=out_channels*in_head, out_channels=features_size, concat=False, heads=1)

      self.skip1 = Linear(input_size, out_channels*in_head)
      self.skip2 = Linear(out_channels*in_head, features_size)

    else:
      # GCN
      self.conv1 = GCNConv(features_size, features_size)
      self.conv2 = GCNConv(features_size, features_size)

      self.preprocess = Linear(input_size, features_size)

      self.relu = LeakyReLU(0.1)
        
  def forward(self, x, edge_index):
    if self.kind == 'gat':
      x1 = self.conv1(x, edge_index)
      x = self.skip1(x)
      x = x + x1
      x = self.relu(x)
      x2 = self.conv2(x, edge_index)
      x = self.skip2(x)
      x = x + x2
      x = self.relu(x)
    else:
      x = self.preprocess(x)
      x1 = self.conv1(x, edge_index)
      x = x + x1
      x2 = self.conv2(x, edge_index)
      x = x + x2
      x = self.relu(x)
       
    return x



class SecondEncoder(torch.nn.Module):
  def __init__(self, input_size, features_size, out_channels=8, in_head=8, dropout=0.5, kind='gcn'):
    super(SecondEncoder, self).__init__()
    self.kind = kind
    if self.kind == 'gat':
      self.conv1 = GATv2Conv(in_channels=input_size, out_channels=out_channels, heads=in_head, dropout=dropout)
      self.conv2 = GATv2Conv(in_channels=out_channels*in_head, out_channels=features_size, concat=False, heads=1)

    else:
      # GCN
      self.conv1 = GCNConv(features_size, features_size)
      self.conv2 = GCNConv(features_size, features_size)

      self.relu = LeakyReLU(0.1)
        
  def forward(self, x, edge_index):
    if self.kind == 'gat':
      x = self.conv1(x, edge_index)
      x = self.relu(x)
      x = self.conv2(x, edge_index)
      x = self.relu(x)
    else:
      x = self.conv1(x, edge_index)
      x = self.conv2(x, edge_index)
      x = self.relu(x)
       
    return x