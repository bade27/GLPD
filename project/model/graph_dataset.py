from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


import os
import torch
from torch.utils.data import Dataset
from project.utils.general_utils import load_pickle



class MetaDataset(Dataset):
  def __init__(self, base_dir):
    self.base_dir = base_dir
    self.nodes_dir = os.path.join(self.base_dir, 'nodes')
    self.variant_dir = os.path.join(self.base_dir, 'variants')
    self.graph_dir = os.path.join(self.base_dir, 'raw')

    self.nodes_name = sorted(os.listdir(self.nodes_dir))
    self.variants_name = sorted(os.listdir(self.variant_dir))
    self.index_names = sorted([f for f in os.listdir(self.graph_dir) if 'graph' in f])
    self.original_names = sorted([f for f in os.listdir(self.graph_dir) if 'original' in f])
    self.y_names = sorted([f for f in os.listdir(self.graph_dir) if 'y' in f])
    self.x_names = sorted([f for f in os.listdir(self.graph_dir) if 'x' in f])

  def __len__(self):
    assert len(self.index_names) == len(self.original_names)
    assert len(self.index_names) == len(self.y_names)
    assert len(self.x_names) == len(self.y_names)
    assert len(self.index_names) == len(self.nodes_name)
    assert len(self.nodes_name) == len(self.variants_name)

    return len(self.index_names)

  def __getitem__(self, idx):
    index_f = os.path.join(self.graph_dir, self.index_names[idx])
    original_f = os.path.join(self.graph_dir, self.original_names[idx])
    y_f = os.path.join(self.graph_dir, self.y_names[idx])
    x_f = os.path.join(self.graph_dir, self.x_names[idx])
    nodes_f = os.path.join(self.nodes_dir, self.nodes_name[idx])
    variants_f = os.path.join(self.variant_dir, self.variants_name[idx])

    edge_index = torch.load(index_f)
    original = torch.load(original_f)
    y = torch.load(y_f)
    x = torch.load(x_f)
    nodes = load_pickle(nodes_f)
    variants = load_pickle(variants_f)


    return x, edge_index, original, y, nodes, variants