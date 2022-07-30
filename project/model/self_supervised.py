import random
import torch
from torch.distributions.categorical import Categorical
from modules.decoder import DecoderMLP
from modules.encoders import FirstEncoder, SecondEncoder
from modules.chooser import Chooser
from modules.info_aggregator import InfoAggregator
from utils.graph_utils import get_forward_star, get_next_activities, node_degree


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

		def forward(self, x, edge_index, original, y, nodes, variants):
			self.chooser.ready_for_training(nodes)
			if self.training:
				_ = self.forward_training(x, edge_index, original, y, nodes, variants)
				return -self.chooser.get_score() # / len(self.chooser.probabilities)
			else:
				return self.inference(x, edge_index, original, y, nodes, variants)

		def forward_training(self, x, edge_index, original, y, nodes, variants):
			embeddings = self.first_encoder(x, edge_index)
			final_places_info = self.aggregator(embeddings, variants, nodes, original, edge_index)
			
			places = set()
			activities = [i for i in range(nodes.index('|')+1)]
			for activity in activities:
				chosen_places = self.chooser(final_places_info, activity, original)
				for chosen_place in chosen_places:
					places.add(chosen_place)

			return places
				

		def inference(self, x, edge_index, original, y, nodes, variants):
			with torch.no_grad():
				places = self.forward_training(x, edge_index, original, y, nodes, variants)
				return places


		