import platform
import os
import argparse
from data_handling.dataset import Dataset
import data_handling.stats as stats
from utils.general_utils import create_dirs
from utils.general_utils import copy_dir
from model.training import Trainer


machine = platform.system()
path_to_windows = lambda path: path.replace('/', '\\')

parser = argparse.ArgumentParser()
parser.add_argument("--generate_data", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model_type", type=str, default="selfsupervised")
parser.add_argument("--optimizer", type=str, default="ADAM")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gnn_type", type=str, default="gcn")
parser.add_argument("--base_dir", type=str, default="/home/linuxpc/MEGAsync/all_data_tesi/data")


if __name__ == '__main__':
    args = parser.parse_args()
    generate_data = args.generate_data
    do_train = args.train
    do_test = args.test
    base_dir = args.base_dir if machine == "Linux" else path_to_windows(args.base_dir)   
    model_type = args.model_type
    optimizer =args.optimizer
    lr = args.lr
    gnn_type = args.gnn_type

    create_dirs([base_dir])
 
    if generate_data:
        copy_before_split = os.path.join(base_dir, '..', "data_before_split")
        copy_after_split = os.path.join(base_dir, '..', "data_after_split")

        # generate data
        dataset = Dataset(base_dir, model_type, random_features=True)
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


    trainer = Trainer(model_type, base_dir, optimizer, lr, gnn_type)

    
    if do_train:
        trainer.train(1, 1, 100)


        

    if do_test:
        trainer.test()