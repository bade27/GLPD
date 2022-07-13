import platform
import os
import argparse
import random
import torch
from tqdm import tqdm
from data_handling.dataset import Dataset
import data_handling.stats as stats
from utils.general_utils import create_dirs
from utils.graph_utils import is_graph_connected
from utils.petri_net_utils import back_to_petri
from utils.pm4py_utils import is_sound, save_petri_net_to_img, save_petri_net_to_pnml
from utils.general_utils import copy_dir
from model import structure as st
from model.training import Trainer
from model.graph_dataset import MetaDataset
from model.self_supervised import SelfSupPredictor


machine = platform.system()
path_to_windows = lambda path: path.replace('/', '\\')

parser = argparse.ArgumentParser()
parser.add_argument("--generate_data", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--base_dir", type=str, default="/home/linuxpc/MEGAsync/all_data_tesi/data")


if __name__ == '__main__':
    args = parser.parse_args()
    generate_data = args.generate_data
    do_train = args.train
    do_test = args.test
    base_dir = args.base_dir if machine == "Linux" else path_to_windows(args.base_dir)   
    checkpoints_dir = os.path.join(base_dir, "..", "checkpoints")
    inference_dir = os.path.join(base_dir, "..", "inference")

    create_dirs([base_dir, checkpoints_dir, inference_dir])
 
    if generate_data:
        copy_before_split = os.path.join(base_dir, '..', "data_before_split")
        copy_after_split = os.path.join(base_dir, '..', "data_after_split")

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

    if do_train:
        num_node_features, features_size, output_size = st.num_node_features, st.features_size, st.output_size

        train = MetaDataset(os.path.join(base_dir, "train_graphs"))
        valid = MetaDataset(os.path.join(base_dir, "validation_graphs"))
        test = MetaDataset(os.path.join(base_dir, "test_graphs"))


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'

        l_train = len(train)
        ii = [n for n in range(l_train)]


        model = SelfSupPredictor(num_node_features, features_size, output_size, 'gcn', device)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        trainer = Trainer(model, optimizer, device)

        epochs = 1

        best_loss = float('+inf')
        best_epoch = 0
        no_epochs_no_improv = 0
        patience = 2

        for epoch in range(epochs):
            elements = [i for i in range(len(train))]
            random.shuffle(elements)

            sum_loss = 0
            best_loss = float('-inf')
            best_epoch = 0

            current_model = None
            for i in tqdm(elements):
                current_model, loss = trainer.train(train[i], 100)
                trainer.model = current_model
                sum_loss += loss
    
            no_epochs = epoch

            if sum_loss < best_loss:
                no_epochs_no_improv = 0
                best_epoch = epoch
                best_loss = sum_loss
            else:
                no_epochs_no_improv += 1

            if epoch > patience and no_epochs_no_improv == patience:
                break
            else:
                torch.save(trainer.model.state_dict(), os.path.join(checkpoints_dir, f"self_supervised_{epoch}.pt"))

            trainer.model.load_state_dict(torch.load(os.path.join(checkpoints_dir, f"self_supervised_{best_epoch}.pt")))

    if do_test:
        model = trainer.model
        model.eval()
        
        logs_dir = os.path.join(base_dir, "test_graphs", "logs")
        logs = sorted(os.listdir(logs_dir))

        sound_nets = 0
        connected = 0
        for i in tqdm(range(len(test))):
          x, _, edge_index, _, nodes, _ = test[i]
        
          mask = trainer.test(test[i])
        
          connected += int(is_graph_connected(edge_index, mask))
        
          result = [int(v) for v in mask]
        
          assert sum(result[:nodes.index('|')+1]) == nodes.index('|')+1
        
          net, im, fm = back_to_petri(edge_index, nodes, result)
          sound_nets += int(is_sound(net, im, fm))
        
          name = str(i)
        
          save_petri_net_to_img(net, im, fm, os.path.join(inference_dir, name + '.png'))
          save_petri_net_to_pnml(net, im, fm, os.path.join(inference_dir, name + '.pnml'))
        
        print(f'number of sound graphs {sound_nets}/{len(test)}')
        print(f'number of connected graphs {connected}/{len(test)}')