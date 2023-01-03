from pathlib import Path
import sys

import torch
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from utils.graph_utils import get_next_nodes, get_prev_nodes
import os
from utils.general_utils import load_pickle, dump_to_pickle
import torch
from tqdm import tqdm
import re

train_dir = "D:\\MEGA\\MEGAsync\\trained_model\\data\\train_graphs\\"
for dn in ["prev", "next"]:
    d = os.path.join(train_dir, dn)
    if not os.path.exists(d):
        os.mkdir(d)

prev_dir = os.path.join(train_dir, "prev")
next_dir = os.path.join(train_dir, "next")

all_nodes_names = sorted([os.path.join(train_dir, "nodes", nname) for nname in  os.listdir(os.path.join(train_dir, "nodes"))])
all_original_names = sorted([os.path.join(train_dir, "raw", nname) for nname in  os.listdir(os.path.join(train_dir, "raw")) if "original" in nname])

l = len(all_nodes_names)
assert len(all_nodes_names) == len(all_original_names)
print(len(all_nodes_names), len(all_original_names))

all_data_names = zip(all_original_names, all_nodes_names)
count = 0
for original_edge_index_name, nodes_name in all_data_names:
    original_edge_index = torch.load(original_edge_index_name)
    nodes = load_pickle(nodes_name)
    next_nodes = get_next_nodes(original_edge_index, nodes)
    prev_nodes = get_prev_nodes(original_edge_index, nodes)

    numb_nodes = int(re.findall(r"\d+", nodes_name)[0])
    numb_or = int(re.findall(r"\d+", original_edge_index_name)[0])

    assert numb_nodes == numb_or

    numb = numb_nodes
    print(f"{numb}, {count}/{l}")
    
    prev_nodes_file_name = os.path.join(prev_dir, "prev_" + str(numb).zfill(4))
    dump_to_pickle(prev_nodes_file_name, prev_nodes)

    next_nodes_file_name = os.path.join(next_dir, "next_" + str(numb).zfill(4))
    dump_to_pickle(next_nodes_file_name, next_nodes)

    count += 1