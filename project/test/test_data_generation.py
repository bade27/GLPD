from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from data_handling.dataset import Dataset
from data_handling import stats

dataset = Dataset("C:\\Users\\matte\\Desktop\\sound_dataset", random_features=True)
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
dataset.generate_dataset(save_networkx=True, save_images=True, save_models=True, visualize_nets=False)
dataset.clean_dataset_and_split()