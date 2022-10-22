from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from data_handling.dataset import Dataset
from data_handling import stats
import os
from utils.general_utils import copy_dir
from model.training import Trainer

# base_dir = "/home/linuxpc/Documenti/TESI/data/"
base_dir = "/home/linuxpc/Documenti/TESI_W_PREV/data/"
model_path = "/home/linuxpc/Documenti/TESI/best_model/self_supervised_97.pt"

copy_before_split = os.path.join(base_dir, '..', "data_before_split")
copy_after_split = os.path.join(base_dir, '..', "data_after_split")

dataset = Dataset(data_dir=base_dir, synth=True, type_of_features="temporal", filename="")
dataset.set_statistics(
    stats.mode, 
    stats.min, 
    stats.max, 
    stats.sequence, 
    stats.choice, 
    stats.parallel, 
    stats.loop, 
    stats.or_gate, 
    stats.silent,
    stats.no_models, 
    stats.no_traces, 
    stats.no_datasets, 
    stats.no_features)

# dataset.generate_dataset(save_images=True, save_models=True, visualize_nets=False)
# copy_dir(base_dir, copy_before_split)
# dataset.clean_dataset_and_split()
# copy_dir(base_dir, copy_after_split)

# trainer = Trainer(base_dir, do_train=True, optimizer_name="SGD", lr=1e-4, gnn_type="gcn", momentum=0.0)
# trainer.train(100, 10, 30)
trainer = Trainer(base_dir, do_test=True, model_path=model_path, gnn_type="gcn", type_of_features="temporal")
trainer.test()
trainer.test(silent_transitions=True)