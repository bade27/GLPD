import torch
from torch.distributions.categorical import Categorical
from modules.decoder import DecoderMLP
from utils.graph_utils import get_forward_star, get_next_activities


class Chooser(torch.nn.Module):
	def __init__(self, input_size, output_size, epsilon=1e-5):
		super(Chooser, self).__init__()
		self.decoder = DecoderMLP(input_size, output_size)
		self.sigmoid = torch.nn.Sigmoid()
		self.probabilities = []
		self.chosen_nodes = []
		self.mask = []
		self.scores = []
		self.epsilon = epsilon


	def get_score(self):
		if len(self.probabilities) == 0:
			return 0
		return torch.concat(self.probabilities, dim=0).sum()

	
	def ready_for_training(self, nodes):
		self.probabilities = []
		self.chosen_nodes = []
		self.mask = ['p' not in n for n in nodes]


	def forward(self, embeddings, activity, original, nodes, nextt):
		next_places = nextt[nodes[activity]]
		to_reach = set()
		for np in next_places:
			for na in nextt[np]:
				to_reach.add(na)

		chosen_places = set()

		pool_of_places = [nodes.index(p) for p in next_places]

		available_places = list(set(pool_of_places).difference(set(self.chosen_nodes)))
		pool_of_places = available_places.copy()

		output = self.decoder(embeddings).squeeze()

		if len(pool_of_places) == 0:
			return chosen_places

		# distribution = self.log_sigmoid(output[pool_of_places])
		
		# assert len(distribution) == len(pool_of_places)

		scores = output[pool_of_places]

		_, ranking = torch.sort(scores, descending=True)

		ranking_map = {pool_of_places[n.item()]:i for i,n in enumerate(ranking)}
		pool_of_places.sort(key = lambda x: ranking_map[x])

		reached = set()
		cnext_places = [plc for plc in nextt[nodes[activity]] if self.mask[nodes.index(plc)]]
		for np in cnext_places:
			for na in nextt[np]:
				reached.add(na)

		idx = 0

		while len(to_reach.difference(reached)) > 0:
			place = pool_of_places[idx]
			if not self.mask[place]:
				self.mask[place] = True
				chosen_places.add(place)			
				cnext_places = [plc for plc in nextt[nodes[activity]] if self.mask[nodes.index(plc)]]
				reached = set()
				for np in cnext_places:
					for na in nextt[np]:
						reached.add(na)
			idx += 1

		if len(chosen_places) > 0:
			probabilities = self.sigmoid(scores)[[pool_of_places.index(place) for place in chosen_places]]
			# print(scores)
			# print(probabilities)
			# print("-"*100)
			for probability in probabilities:
				# print("prob ", probability-self.epsilon)
				# print("sum ", torch.sum(probabilities))
				# print((probability-self.epsilon) / torch.sum(probabilities))
				# self.probabilities.append(
				# 	torch.log(
				# 		(probability) / (torch.sum(probabilities)+self.epsilon) + self.epsilon
				# 		).unsqueeze(-1)
				# 	)
				self.probabilities.append(torch.log(probability+self.epsilon).unsqueeze(-1))
		return chosen_places