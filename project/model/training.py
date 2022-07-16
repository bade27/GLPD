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

from utils.graph_utils import is_graph_connected
from utils.petri_net_utils import back_to_petri
from utils.pm4py_utils import is_sound, save_petri_net_to_img, save_petri_net_to_pnml


difference = lambda x, y: abs((-torch.sum(x)-torch.sum(y)).item())
optimizer = {"ADAM":torch.optim.Adam, "SGD":torch.optim.SGD}


class Trainer():
	def __init__(self, model_type, base_dir, optimizer_name, lr, gnn_type, criterion=None):
		self.optimizer_name = optimizer_name
		self.lr = lr
		self.gnn_type = gnn_type
		self.model_type = model_type
		self.base_dir = base_dir
		self.model = None
		self.device = None
		self.optimizer = None
		self.criterion = criterion

		self.train_dataset = MetaDataset(os.path.join(self.base_dir, "train_graphs"))
		self.valid_dataset = MetaDataset(os.path.join(self.base_dir, "validation_graphs"))
		self.test_dataset = MetaDataset(os.path.join(self.base_dir, "test_graphs"))

		random.seed(1234)

		self.checkpoints_dir = os.path.join(base_dir, "..", "checkpoints")
		self.inference_dir = os.path.join(base_dir, "..", "inference")
		create_dirs([self.checkpoints_dir, self.inference_dir])
		
	def set_model(self, model):
		self.model = model
		
	def get_model(self):
		return self.model
		
		
	def train(self, epochs, patience, max_runs=None):
		num_node_features, features_size, output_size = st.num_node_features, st.features_size, st.output_size
		
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.device = 'cpu' 
		
		if self.model_type == "supervised":
			self.model = SupervisedPredictor(num_node_features, features_size, output_size, self.gnn_type, self.device)
		else:
			self.model = SelfSupPredictor(num_node_features, features_size, output_size, self.gnn_type, self.device)
			
		self.model = self.model.to(self.device)
		self.optimizer = optimizer[self.optimizer_name](self.model.parameters(), self.lr)
		
		
		if self.model_type == 'supervised':
			self.train_supervised(epochs, patience) 
		else:
			self.train_self_supervised(epochs, patience, max_runs)    
			
			
	def train_self_supervised(self, epochs, patience, max_runs):
		best_loss = float('+inf')
		best_epoch = 0
		no_epochs_no_improv = 0
		
		for epoch in range(epochs):
			elements = [i for i in range(len(self.train_dataset))]
			random.shuffle(elements)
			
			sum_loss = 0
			best_loss = float('-inf') # delete this
			best_epoch = 0 # delete this
			
			for i in tqdm(elements):
				x, edge_index, original, y, nodes, variants = self.train_dataset[i]
				x = x.to(self.device)
				edge_index = edge_index.to(self.device)
				
				cumulative_loss = []
				prev_prediction = torch.zeros(len(nodes), 1, device=self.device)
				prev_difference = float('+inf')
				
				self.model.train()
				self.optimizer.zero_grad()
				prediction = self.model(x, edge_index, original, y, nodes, variants)
				current_difference = difference(prediction, prev_prediction)
				loss = -torch.sum(prediction)
				loss.backward()
				self.optimizer.step()
				
				cumulative_loss.append(loss.item())
				max_runs -= 1
				
				while abs(prev_difference-current_difference) > 1e-5 and max_runs > 0:
					prev_difference = current_difference
					prev_prediction = prediction
					self.model.train()
					self.optimizer.zero_grad()
					prediction = self.model(x, edge_index, original, y, nodes, variants)
					loss = -torch.sum(prediction)
					loss.backward()
					self.optimizer.step()
					current_difference = difference(prediction, prev_prediction)
					max_runs -= 1
					
				sum_loss += np.mean(cumulative_loss)
				
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
			print(f"epoch {epoch} - loss: {mean_loss[-1]} - acc: {mean_acc[-1]}")

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

		for i in tqdm(range(len(self.valid_dataset))):
			x, edge_index, original, y, nodes, variants = self.valid_dataset[i]

			x = x.to(self.device)
			y = y.to(self.device)
			edge_index = edge_index.to(self.device)

			mask = self.model(x, edge_index, original, y, nodes, variants)

			result = [int(v) for v in mask]

			assert sum(result[:nodes.index('|')+1]) == nodes.index('|')+1

			accuracy = sum(result[nodes.index('|')+1:]) / (len(result)-nodes.index('|')+1)

		return accuracy

	
	def test(self):
		self.model.eval()

		sound_nets = 0
		connected = 0
		for i in tqdm(range(len(self.test_dataset))):
			x, edge_index, original, y, nodes, variants = self.test_dataset[i]

			x = x.to(self.device)
			edge_index = edge_index.to(self.device)

			mask = self.model(x, edge_index, original, y, nodes, variants)

			connected += int(is_graph_connected(original, mask))

			result = [int(v) for v in mask]

			assert sum(result[:nodes.index('|')+1]) == nodes.index('|')+1

			net, im, fm = back_to_petri(original, nodes, result)

			sound_nets += int(is_sound(net, im, fm))

			name = str(i)

			save_petri_net_to_img(net, im, fm, os.path.join(self.inference_dir, name + '.png'))
			save_petri_net_to_pnml(net, im, fm, os.path.join(self.inference_dir, name + '.pnml'))

		print(f'number of sound graphs {sound_nets}/{len(self.test_dataset)}')
		print(f'number of connected graphs {connected}/{len(self.test_dataset)}')