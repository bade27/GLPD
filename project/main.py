import platform
import os
from data_handling.dataset import Dataset
import data_handling.stats as stats
from utils.general_utils import copy_dir


if __name__ == '__main__':
    machine = platform.system()
    
    if machine == 'Linux':
        base_dir = "/home/linuxpc/MEGAsync/all_data_tesi/data"
        copy_before_split = "/home/linuxpc/MEGAsync/all_data_tesi/data_before_split"
        copy_after_split = "/home/linuxpc/MEGAsync/all_data_tesi/data_after_split"
    else:
        base_dir = "D:\\MEGA\\MEGAsync\\all_data_tesi\\data"
        copy_before_split = "D:\\MEGA\\MEGAsync\\all_data_tesi\\data_before_split"
        copy_after_split = "D:\\MEGA\\MEGAsync\\all_data_tesi\\data_after_split"

    # generate data
    dataset = Dataset(base_dir, random_features=True)
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

    copy_dir(base_dir, copy_before_split)

    dataset.clean_dataset_and_split()

    copy_dir(base_dir, copy_after_split)

