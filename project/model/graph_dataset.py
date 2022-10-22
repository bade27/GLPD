from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


import os
import torch
from torch.utils.data import Dataset
from utils.general_utils import load_pickle



class MetaDataset(Dataset):
  def __init__(self, base_dir, device, type_of_features="temporal"):
    self.device = device

    # dirs
    self.base_dir = base_dir
    self.type_of_features = type_of_features
    self.nodes_dir = os.path.join(self.base_dir, 'nodes')
    self.variant_dir = os.path.join(self.base_dir, 'variants')
    self.graph_dir = os.path.join(self.base_dir, 'raw')
    self.order_dir = os.path.join(self.base_dir, 'order')
    self.next_dir = os.path.join(self.base_dir, 'next')
    self.prev_dir = os.path.join(self.base_dir, 'prev')

    # list of names
    self.nodes_names = sorted(os.listdir(self.nodes_dir))
    self.variants_names = sorted(os.listdir(self.variant_dir))
    self.index_names = sorted([f for f in os.listdir(self.graph_dir) if 'graph' in f])
    self.original_names = sorted([f for f in os.listdir(self.graph_dir) if 'original' in f])
    if self.type_of_features == "temporal" or self.type_of_features== "random":
      self.x_names = sorted([f for f in os.listdir(self.graph_dir) if 'x' in f])
    else:
      self.x_names = [f for f in os.listdir(os.path.join(self.graph_dir, "features"))]
      self.features_size = torch.load(os.path.join(self.graph_dir, "features", self.x_names[0])).shape[1]
    self.order_names = sorted(os.listdir(self.order_dir))
    self.next_names = sorted(os.listdir(self.next_dir))
    self.prev_names = sorted(os.listdir(self.prev_dir))

    # data
    self.edge_indices_list = []
    for index_name in self.index_names:
      index_f = os.path.join(self.graph_dir, index_name)
      edge_index = torch.load(index_f)
      self.edge_indices_list.append(edge_index)

    self.originals_list = []
    for original_name in self.original_names:
      original_f = os.path.join(self.graph_dir, original_name)
      original = torch.load(original_f)
      self.originals_list.append(original)

    self.nodes_list = []
    for nodes_name in self.nodes_names:
      nodes_f = os.path.join(self.nodes_dir, nodes_name)
      nodes = load_pickle(nodes_f)
      self.nodes_list.append(nodes)
    
    self.variants_list = []
    for variants_name in self.variants_names:
      variants_f = os.path.join(self.variant_dir, variants_name)
      variants = load_pickle(variants_f)
      self.variants_list.append(variants)

    self.xs_list = []
    for idx, x_name in enumerate(self.x_names):
      x_f = os.path.join(self.graph_dir, x_name)
      if self.type_of_features == "temporal" or self.type_of_features== "random":
        x = torch.load(x_f)
      else:
        x = torch.load(x_f)
        alphabets = [(chr(ord('a')+i)) for i in range(26)]
        indices = [0] + [alphabets.index(n) for n in self.nodes_list[idx]] + [27]
        x = x[indices,:]
      self.xs_list.append(x)

    self.order_list = []
    for order_name in self.order_names:
      order_f = os.path.join(self.order_dir, order_name)
      order = load_pickle(order_f)
      self.order_list.append(order)

    self.next_list = []
    for next_name in self.next_names:
      next_f = os.path.join(self.next_dir, next_name)
      nextt = load_pickle(next_f)
      self.next_list.append(nextt)
    
    self.prev_list = []
    for prev_name in self.prev_names:
      prev_f = os.path.join(self.prev_dir, prev_name)
      prev = load_pickle(prev_f)
      self.prev_list.append(prev)
  

  def __len__(self):
    assert len(self.index_names) == len(self.original_names)
    assert len(self.index_names) == len(self.x_names)
    assert len(self.index_names) == len(self.nodes_names)
    assert len(self.nodes_names) == len(self.variants_names)
    assert len(self.nodes_names) == len(self.order_names)
    assert len(self.order_names) == len(self.next_names)
    assert len(self.prev_names) == len(self.next_names)

    return len(self.index_names)

  def __getitem__(self, idx):
    x = (self.xs_list[idx]).to(self.device)
    edge_index = (self.edge_indices_list[idx]).to(self.device)
    original = (self.originals_list[idx]).to(self.device)
    nodes = self.nodes_list[idx]
    variants = self.variants_list[idx]
    order = self.order_list[idx]
    nextt = self.next_list[idx]
    prev = self.prev_list[idx]

    return x, edge_index, original, nodes, variants, order, nextt, prev