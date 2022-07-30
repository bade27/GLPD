import random
import torch
from modules.encoders import SecondEncoder
from utils.graph_utils import get_forward_star


class InfoAggregator(torch.nn.Module):
    def __init__(self, embedding_size, device, kind):
        super(InfoAggregator, self).__init__()
        self.embedding_size = embedding_size
        self.cat_embedding_size = self.embedding_size*3
        self.device = device
        self.encoder = SecondEncoder(embedding_size, embedding_size, kind=kind)

    def forward(self, embeddings, variants, nodes, original, edge_index):
        final_places_info = torch.zeros(len(nodes), self.cat_embedding_size, device=self.device)

        random.shuffle(variants)

        prev_embeddings = embeddings

        for variant in variants:
            embeddings = prev_embeddings
            activities = variant
            places_info = torch.zeros(len(nodes), self.cat_embedding_size, device=self.device)
            prev_embedding_summary = 0

            for i, activity in enumerate(activities):
                activity_idx = nodes.index(activity)
                all_followers, _ = get_forward_star(original, activity_idx)

                if len(all_followers) == 0:
                    continue

                in_variant_followers = {}
                for next_activity in activities[i+1:]:
                    next_activity_idx = nodes.index(next_activity)
                    for follower in all_followers:
                        next_next, _ = get_forward_star(original, follower)
                        if next_activity_idx in next_next:
                            if follower not in in_variant_followers:
                                in_variant_followers[follower] = set()
                            in_variant_followers[follower].add(next_activity_idx)

                considered_places = set()
                for place, destination in in_variant_followers.items():
                    src = activity_idx
                    src_f = embeddings[src].clone()
                    place_f = embeddings[place]
					
                    dst_f = embeddings[list(destination)[0]].clone()
                    if len(destination) > 1:
                        for l in range(1, len(destination)):
                            dst_f += embeddings[list(destination)[l]].clone()
					
                    src_f = src_f.unsqueeze(0)
                    dst_f = dst_f.unsqueeze(0)
                    place_f = place_f.unsqueeze(0)
					
                    embeddings_cat = torch.cat((place_f, src_f, dst_f),1)

                    places_info[place] += embeddings_cat.squeeze()
					
                    prev_embedding_summary += places_info[place]
                    considered_places.add(place)
					
                if len(considered_places) > 0:
                    places_info[list(considered_places)] += prev_embedding_summary
							
                embeddings = self.encoder(embeddings, edge_index)
					
                final_places_info += places_info

        return final_places_info