import torch
from torch.distributions.categorical import Categorical

from utils.graph_utils import get_forward_star, get_next_activities


class Chooser(torch.nn.Module):
	def __init__(self, device, epsilon=1e-5):
		super(Chooser, self).__init__()
		self.device = device
		self.sigmoid = torch.nn.Sigmoid()
		self.epsilon = epsilon


	def get_probabilities(self):
		return torch.log(self.probabilities[1:])

	
	def ready_for_training(self, nodes):
		self.probabilities = torch.zeros((1,1), device=self.device)
		self.chosen_nodes = set()
		self.mask = ['p' not in n for n in nodes]


	def forward(self, place_scores, activity, original, nodes, nextt):
		next_places = nextt[nodes[activity]]
		to_reach = set()
		for np in next_places:
			for na in nextt[np]:
				to_reach.add(na)

		chosen_places = set()

		pool_of_places = [nodes.index(p) for p in next_places]

		available_places = list(set(pool_of_places).difference(set(self.chosen_nodes)))
		pool_of_places = available_places.copy()

		if len(pool_of_places) == 0:
			return chosen_places

		weights = torch.tensor([len(nextt[nodes[plc]]) for plc in pool_of_places], device=self.device).float()
		weights /= weights.sum()

		_, ranking = torch.sort(place_scores[pool_of_places]*weights, descending=True)

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
				current_score = self.sigmoid(place_scores[place])
				self.probabilities = torch.cat(
					[self.probabilities, torch.reshape(current_score, (1, 1))],  dim=0)
				self.mask[place] = True
				chosen_places.add(place)
				self.chosen_nodes.add(place)
				cnext_places = [plc for plc in nextt[nodes[activity]] if self.mask[nodes.index(plc)]]
				reached = set()
				for np in cnext_places:
					for na in nextt[np]:
						reached.add(na)
			idx += 1

		return chosen_places