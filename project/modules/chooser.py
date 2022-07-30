import torch
from torch.distributions.categorical import Categorical
from modules.decoder import DecoderMLP
from utils.graph_utils import get_forward_star, get_next_activities, node_degree


class Chooser(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(Chooser, self).__init__()
		self.decoder = DecoderMLP(input_size, output_size)
		self.log_softmax = torch.nn.LogSoftmax(dim=0)
		self.log_sigmoid = torch.nn.LogSigmoid()
		self.probabilities = []
		self.chosen_nodes = []
		self.mask = []


	def get_score(self):
		return torch.concat(self.probabilities, dim=0).sum()

	
	def ready_for_training(self, nodes):
		self.probabilities = []
		self.chosen_nodes = []
		self.mask = ['p' not in n for n in nodes]


	def forward(self, embeddings, activity, original):
		connections = get_next_activities(original, activity)

		chosen_places = set()

		pool_of_places, _ = get_forward_star(original, activity)

		output = self.decoder(embeddings).squeeze()

		available_places = list(set(pool_of_places).difference(set(self.chosen_nodes)))
		pool_of_places = available_places.copy()

		if len(pool_of_places) == 0:
			return chosen_places

		distribution = self.log_sigmoid(output[pool_of_places])
		
		assert len(distribution) == len(pool_of_places)


		_, ranking = torch.sort(distribution, descending=True)

		ranking_map = {pool_of_places[n.item()]:i for i,n in enumerate(ranking)}
		pool_of_places.sort(key = lambda x: ranking_map[x])

		while True:
			discovered_next = get_next_activities(original, activity, self.mask)
			if discovered_next != connections:
				for position in ranking:
					place = available_places[position.item()]
					if not self.mask[place]:
						self.mask[place] = True
						chosen_places.add(place)
						probability = distribution[position.item()]
						self.probabilities.append(probability.unsqueeze(-1))
						break
			else:
				break
		return chosen_places
		
		# if len(pool_of_places) == 1:
		# 	sampled_place = pool_of_places[0]
		# 	self.mask[sampled_place] = True
		# 	self.chosen_nodes.append(sampled_place)
		# 	self.probabilities.append(distribution[pool_of_places.index(sampled_place)].unsqueeze(-1))
		# elif len(pool_of_places) > 1:
		# 	sampler = Categorical(distribution)
	
		# 	sampled_places = set()
		# 	sampled_places = sampled_places.union(set(self.chosen_nodes))
		# 	sampled_indices = set()
		# 	while True:
		# 		discovered_next = get_next_activities(original, activity, self.mask)
		# 		if connections != discovered_next:
		# 			sampled_idx = None
		# 			sampled_place = None
		# 			count = 0
		# 			while sampled_place in sampled_places.union({None}):
		# 				sampled_idx = sampler.sample()
		# 				sampled_place = pool_of_places[sampled_idx]
		# 				count += 1
		# 				if count > 1000:
		# 					break
		# 			print(count)
		# 			self.mask[sampled_place] = True
		# 			sampled_places.add(sampled_place)
		# 			sampled_indices.add(sampled_idx)
		# 			self.chosen_nodes.append(sampled_place)
		# 			self.probabilities.append(distribution[sampled_idx].unsqueeze(-1))
		# 		else:
		# 			break