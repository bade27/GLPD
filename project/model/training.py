from pathlib import Path
import shutil
import sys

from scipy.stats import moment


path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import pm4py
import os
import numpy as np
import torch
from tqdm import tqdm
import model.structure as st
from model.graph_dataset import MetaDataset
from model.self_supervised import SelfSupPredictor
import random
import matplotlib.pyplot as plt
from model.supervised import SupervisedPredictor
from utils.general_utils import create_dirs
from utils.general_utils import load_pickle
from utils.graph_utils import is_graph_connected, add_silent_transitions
from utils.petri_net_utils import back_to_petri
from utils.pm4py_utils import is_sound, save_petri_net_to_img, save_petri_net_to_pnml


difference = lambda x, y: abs((-torch.sum(x)-torch.sum(y)).item())
optimizer = {"ADAM":torch.optim.Adam, "SGD":torch.optim.SGD}


class Trainer():
	def __init__(self, base_dir, optimizer_name, lr, gnn_type, type_of_features="temporal"):
		self.optimizer_name = optimizer_name
		self.lr = lr
		self.gnn_type = gnn_type
		self.base_dir = base_dir

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"

		print(f"device: {self.device}")
		print("loading datasets...")
		self.train_dataset = MetaDataset(
			os.path.join(self.base_dir, "train_graphs"), device=self.device, type_of_features=type_of_features)
		print("train dataset loaded")
		self.test_dataset = MetaDataset(
			os.path.join(self.base_dir, "test_graphs"), device=self.device, type_of_features=type_of_features)
		print("test dataset loaded")
		
		random.seed(1234)

		self.checkpoints_dir = os.path.join(base_dir, "..", "checkpoints")
		self.best_model_dir = os.path.join(base_dir, "..", "best_model")
		self.inference_dir = os.path.join(base_dir, "test_graphs", "inference")
		create_dirs([self.checkpoints_dir, self.best_model_dir, self.inference_dir])

		num_node_features, features_size = st.num_node_features, st.features_size
			
		output_size = st.output_size_self_sup
		self.model = SelfSupPredictor(num_node_features, features_size, output_size, self.gnn_type, self.device)
			
		self.model = self.model.to(self.device)
		# self.optimizer = optimizer[self.optimizer_name](self.model.parameters(), self.lr)
		self.optimizer = optimizer["SGD"](self.model.parameters(), 0.1, momentum=0.9)

		
	def set_model(self, model):
		self.model = model
		
	def get_model(self):
		return self.model
		
		
	def train(self, epochs, patience, max_runs, theta=1e-3):
		best_loss = float('+inf')
		best_epoch = 0
		no_epochs = 0
		no_epochs_no_improv = 0

		epoch_loss = []
		mean_numer_of_runs = []
		
		for epoch in range(epochs):
			elements = [i for i in range(len(self.train_dataset))][:100]
			
			sum_loss = 0
			no_epoch_runs = []
			
			random.shuffle(elements)
			for i in tqdm(elements):
				x, edge_index, original, nodes, variants, order, nextt = self.train_dataset[i]
				
				cumulative_loss = []
				no_runs = 0
				prev_prob = 0
				
				for run in tqdm(range(max_runs)):
					self.model.train()
					self.optimizer.zero_grad()
					score = self.model(x, edge_index, original, nodes, variants, order, nextt)
					loss = score
					loss.backward()
					self.optimizer.step()
					cumulative_loss.append(score.item())
					no_runs += 1
					if abs(score - prev_prob) < theta and run > 0:
						break
					prev_prob = loss.item()

				sum_loss += np.mean(cumulative_loss)
			
				no_epoch_runs.append(no_runs)

			mean_numer_of_runs.append(np.mean(no_epoch_runs))
			epoch_loss.append(sum_loss)

			print(f"epoch {epoch+1} - loss: {sum_loss}")

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
				torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, f"self_supervised_{epoch}.pt"))
				
		print(f"total epochs passed {no_epochs+1}")
		print(f"best poch: {best_epoch+1}")
		shutil.copy(
			os.path.join(self.checkpoints_dir, f"self_supervised_{best_epoch}.pt"), 
			os.path.join(self.best_model_dir, f"self_supervised_{best_epoch}.pt"))
		self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, f"self_supervised_{best_epoch}.pt")))
		
		plt.plot([i for i in range(no_epochs+1)], epoch_loss)
		plt.ylabel("Loss")
		plt.xlabel("Epochs")
		plt.savefig(os.path.join(self.best_model_dir, "loss.png"))


	def test(self, silent_transitions=False):
		if len(os.listdir(self.best_model_dir)) > 0:
			file = os.listdir(self.best_model_dir)[0]
			self.model.load_state_dict(torch.load(os.path.join(self.best_model_dir, file)))
		
		img_dir = os.path.join(self.inference_dir, "images")
		pnml_dir = os.path.join(self.inference_dir, "pnml")
		alpha_relations_dir = os.path.join(self.base_dir, "test_graphs", "alpha_relations")

		create_dirs([img_dir, pnml_dir])

		idxes = [int(name.split('_')[1].split('.')[0]) for name in os.listdir(alpha_relations_dir)]
		assert len(idxes) == len(self.test_dataset)

		alpha_relations_names = sorted(os.listdir(alpha_relations_dir))

		self.model.eval()

		sound_nets = 0
		connected = 0

		for i in tqdm(range(len(self.test_dataset))):
			x, edge_index, original, nodes, variants, order, nextt = self.test_dataset[i]

			places = self.model(x, edge_index, original, nodes, variants)

			mask = ['p' not in n for n in nodes]
			for place in places:
				mask[place] = True

			connected += int(is_graph_connected(original, mask))

			assert sum(mask[:nodes.index('|')+1]) == nodes.index('|')+1

			if silent_transitions:
				alpha_relations = load_pickle(os.path.join(alpha_relations_dir, alpha_relations_names[i]))
				new_edge_index, new_nodes = add_silent_transitions(original, mask, nodes, alpha_relations)

				for _ in range(len(new_nodes) - len(nodes)):
					mask.append(True)

				excluded = set([i for i in range(len(mask)) if not mask[i]])
				to_check = set()
				for item in excluded:
					for h in range(len(edge_index[0])):
						if edge_index[0][h].item() == item:
							to_check.add(edge_index[1][h].item())
						elif edge_index[1][h].item() == item:
							to_check.add(edge_index[0][h].item())
				for t in to_check:
					if "silent" in nodes[t]:
						excluded.add(t)

				for k in range(len(mask)):
					if i in excluded:
						mask[k] = False
			else:
				new_edge_index = original
				new_nodes = nodes


			net, im, fm = back_to_petri(new_edge_index, new_nodes, mask)
			sound_nets += int(is_sound(net, im, fm))
			name = "model_" + str(idxes[i])

			save_petri_net_to_img(net, im, fm, os.path.join(img_dir, name + '.png'))
			save_petri_net_to_pnml(net, im, fm, os.path.join(pnml_dir, name + '.pnml'))

		print(f'number of sound graphs {sound_nets}/{len(self.test_dataset)}')
		print(f'number of connected graphs {connected}/{len(self.test_dataset)}')