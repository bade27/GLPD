from tabnanny import check
import torch
from modules.encoders import FirstEncoder
from modules.chooser import Chooser
from modules.info_aggregator import InfoAggregator


class SelfSupPredictor(torch.nn.Module):
	def __init__(self, num_node_features, embedding_size, output_size, encoder, device): # delete device
		super(SelfSupPredictor, self).__init__()
		self.num_node_features = num_node_features
		self.embedding_size = embedding_size
		self.cat_embedding_size = 3*self.embedding_size
		self.output_size = output_size
		self.device = device
		self.first_encoder = FirstEncoder(num_node_features, embedding_size, kind=encoder)
		self.aggregator = InfoAggregator(embedding_size, device, kind=encoder)
		self.chooser = Chooser(embedding_size*3, output_size)

	def forward(self, x, edge_index, original, nodes, variants):
		self.chooser.ready_for_training(nodes)
		if self.training:
			_ = self.forward_training(x, edge_index, original, nodes, variants)
			return -self.chooser.get_score() # / len(self.chooser.probabilities)
		else:
			return self.inference(x, edge_index, original, nodes, variants)

	def forward_training(self, x, edge_index, original, nodes, variants):
		embeddings = self.first_encoder(x, edge_index)
		final_places_info = self.aggregator(embeddings, variants, nodes, original, edge_index)
		
		places = set()
		# activities = [i for i in range(nodes.index('|')+1)]
		activities = self.get_activity_order(original, nodes)

		check_found_activities = set(activities)
		actual_activities = set([i for i in range(nodes.index('|')+1)])
		assert check_found_activities == actual_activities

		for activity in activities:
			chosen_places = self.chooser(final_places_info, activity, original)
			for chosen_place in chosen_places:
				places.add(chosen_place)

		return places
			

	def inference(self, x, edge_index, original, nodes, variants):
		with torch.no_grad():
			places = self.forward_training(x, edge_index, original, nodes, variants)
			return places


	def get_activity_order(edge_index, nodes):
		order = []
		queue = min(edge_index)
		visited = set()

		while queue:
			current = queue.pop(0)
			if current not in visited:
				order.append(current)
				for i in range(len(edge_index[0])):
					if edge_index[0][i].item() == current:
						queue.append(edge_index[1][i].item())
				visited.add(current)
		
		return [i for i in order if i <= nodes.index('|')]
