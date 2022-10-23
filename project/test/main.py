from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from data_handling.dataset import Dataset
from data_handling import stats
import os
from utils.general_utils import copy_dir
from model.training import Trainer
from model import inference

# base_dir = "/home/linuxpc/Documenti/TESI/data/"
base_dir = "C:\\Users\\matte\\Desktop\\MY_PROJECT_DATASET_FINAL\\data\\"
best_model_dir = os.path.join(base_dir, "..", "best_model")

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

dataset.generate_dataset(save_images=True, save_models=True, visualize_nets=False)
copy_dir(base_dir, copy_before_split)
dataset.clean_dataset_and_split()
copy_dir(base_dir, copy_after_split)

trainer = Trainer(base_dir, do_train=True, optimizer_name="SGD", lr=1e-4, gnn_type="gcn", momentum=0.0)
trainer.train(100, 10, 30)

model_path = os.path.join(best_model_dir, os.listdir(best_model_dir)[0])
trainer = Trainer(base_dir, do_test=True, model_path=model_path, gnn_type="gcn", type_of_features="temporal")
trainer.test()
trainer.test(silent_transitions=True)

real_logs_dir = "D:\\MEGA\\MEGAsync\\my_project\\"
for folder in os.listdir(real_logs_dir):
    log_filename = [file for file in os.listdir(os.path.join(real_logs_dir, folder)) if "xes" in file]
    inference.infer(os.path.join(real_logs_dir, folder), log_filename, model_path, silent_transitions=True, preprocessing=True)