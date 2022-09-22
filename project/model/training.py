from pathlib import Path
import shutil
import sys


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
from model.supervised import SupervisedPredictor
from utils.general_utils import create_dirs
from utils.general_utils import load_pickle
from utils.graph_utils import is_graph_connected, add_silent_transitions
from utils.petri_net_utils import back_to_petri
from utils.pm4py_utils import is_sound, save_petri_net_to_img, save_petri_net_to_pnml


difference = lambda x, y: abs((-torch.sum(x)-torch.sum(y)).item())
optimizer = {"ADAM":torch.optim.Adam, "SGD":torch.optim.SGD}


class Trainer():
	def __init__(self, model_type, base_dir, optimizer_name, lr, gnn_type, criterion=None, random_features=True):
		self.optimizer_name = optimizer_name
		self.lr = lr
		self.gnn_type = gnn_type
		self.model_type = model_type
		self.base_dir = base_dir
		self.model = None
		self.device = None
		self.optimizer = None
		self.criterion = criterion

		self.train_dataset = MetaDataset(os.path.join(self.base_dir, "train_graphs"), random_features=random_features)
		self.valid_dataset = MetaDataset(os.path.join(self.base_dir, "validation_graphs"), random_features=random_features)
		self.test_dataset = MetaDataset(os.path.join(self.base_dir, "test_graphs"), random_features=random_features)

		random.seed(1234)

		self.checkpoints_dir = os.path.join(base_dir, "..", "checkpoints")
		self.best_model_dir = os.path.join(base_dir, "..", "best_model")
		self.inference_dir = os.path.join(base_dir, "test_graphs", "inference")
		create_dirs([self.checkpoints_dir, self.best_model_dir, self.inference_dir])

		num_node_features, features_size = st.num_node_features, st.features_size
		
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.device = 'cpu'

		
		if self.model_type == "supervised":
			output_size = st.output_size_sup
			self.model = SupervisedPredictor(num_node_features, features_size, output_size, self.gnn_type, self.device)
		else:
			output_size = st.output_size_self_sup
			self.model = SelfSupPredictor(num_node_features, features_size, output_size, self.gnn_type, self.device)
			
		self.model = self.model.to(self.device)
		self.optimizer = optimizer[self.optimizer_name](self.model.parameters(), self.lr)

		
	def set_model(self, model):
		self.model = model
		
	def get_model(self):
		return self.model
		
		
	def train(self, epochs, patience, max_runs=1):	
		if self.model_type == 'supervised':
			self.train_supervised(epochs, patience) 
		else:
			self.train_self_supervised(epochs, patience, max_runs)    
			
			
	def train_self_supervised(self, epochs, patience, max_runs):
		best_loss = float('+inf')
		best_epoch = 0
		no_epochs_no_improv = 0

		epoch_loss = []
		
		for epoch in range(epochs):
			elements = [i for i in range(len(self.train_dataset))]
			
			sum_loss = 0
			
			random.shuffle(elements)
			for i in tqdm(elements):
				x, edge_index, original, nodes, variants = self.train_dataset[i]
				x = x.to(self.device)
				edge_index = edge_index.to(self.device)
				
				cumulative_loss = []
				prev_prob = 0
				
				while True:
					self.model.train()
					self.optimizer.zero_grad()
					score = self.model(x, edge_index, original, nodes, variants)
					# print(score)
					# print('*'*50)
					loss = score
					loss.backward()
					self.optimizer.step()

					cumulative_loss.append(score.item())

					if abs(score - prev_prob) < 1e-4:
						break
					
					max_runs -= 1
					if max_runs < 0:
						break

				sum_loss += np.mean(cumulative_loss)
			
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
				
		print(f"total epochs passed {no_epochs}")
		print(f"best poch: {best_epoch}")
		shutil.copy(
			os.path.join(self.checkpoints_dir, f"self_supervised_{best_epoch}.pt"), 
			os.path.join(self.best_model_dir, f"self_supervised_{best_epoch}.pt"))
		self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, f"self_supervised_{best_epoch}.pt")))


	def train_supervised(self, epochs, patience):
		max_accuracy = 0
		best_epoch = 0
		no_epochs_no_improv = 0

		mean_loss = []
		mean_acc = []
		
		for epoch in range(epochs):
			epoch_loss = []

			elements = [i for i in range(len(self.train_dataset))]
			random.shuffle(elements)
			
			for i in tqdm(elements):
				x, edge_index, original, y, nodes, variants = self.train_dataset[i]
				x = x.to(self.device)
				y = y.to(self.device)
				edge_index = edge_index.to(self.device)
				
				self.model.train()
				self.optimizer.zero_grad()
				prediction = self.model(x, edge_index, original, y, nodes, variants)
				loss = self.criterion(prediction, y)
				loss.backward()
				self.optimizer.step()

				epoch_loss.append(loss.item())
				
			accuracy = self.validate()

			mean_loss.append(np.mean(epoch_loss))
			mean_acc.append(accuracy)
			print(f"epoch {epoch+1} - loss: {mean_loss[-1]} - acc: {mean_acc[-1]}")

			no_epochs = epoch
			
			if accuracy > max_accuracy:
				no_epochs_no_improv = 0
				best_epoch = epoch
				max_accuracy = accuracy
			else:
				no_epochs_no_improv += 1
				
			if epoch > patience and no_epochs_no_improv == patience:
				break
			else:
				torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, f"supervised_{epoch}.pt"))
				
		print(f"total epochs passed {no_epochs}")
		print(f"best poch: {best_epoch}")
		self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, f"supervised_{best_epoch}.pt")))

  
	def validate(self):
		self.model.eval()

		accuracies = []

		for i in tqdm(range(len(self.valid_dataset))):
			x, edge_index, original, y, nodes, variants = self.valid_dataset[i]

			x = x.to(self.device)
			y = y.to(self.device)
			edge_index = edge_index.to(self.device)

			mask = self.model(x, edge_index, original, y, nodes, variants)

			result = [int(v) for v in mask]

			# assert sum(result[:nodes.index('|')+1]) == nodes.index('|')+1

			print(result)

			accuracy = sum(result[nodes.index('|')+1:]) / (len(result)-nodes.index('|')+1)
			accuracies.append(accuracy)

		return np.mean(accuracies)

	
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
			x, edge_index, original, nodes, variants = self.test_dataset[i]

			x = x.to(self.device)
			edge_index = edge_index.to(self.device)

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
			name = self.model_type + '_' + str(idxes[i])

			save_petri_net_to_img(net, im, fm, os.path.join(img_dir, name + '.png'))
			save_petri_net_to_pnml(net, im, fm, os.path.join(pnml_dir, name + '.pnml'))

		print(f'number of sound graphs {sound_nets}/{len(self.test_dataset)}')
		print(f'number of connected graphs {connected}/{len(self.test_dataset)}')