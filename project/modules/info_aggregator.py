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

    def forward(self, embeddings, variants, nodes, original, edge_index, order, nextt):
        final_places_info = torch.zeros(len(nodes), self.cat_embedding_size, device=self.device)
        places_temp_info = {}
        random.shuffle(variants)

        # prev_embeddings = embeddings

        for variant in variants:
            # embeddings = prev_embeddings
            activities = variant
            places_info = torch.zeros(len(nodes), self.cat_embedding_size, device=self.device)
            prev_embedding_summary = 0

            in_variant_places_temp_info = {}

            for ii, activity in enumerate(activities):
                activity_idx = nodes.index(activity)
                # all_followers, _ = get_forward_star(original, activity_idx)

                # if len(all_followers) == 0:
                #     continue

                # in_variant_followers = {}
                # for next_activity in activities[i+1:]:
                #     next_activity_idx = nodes.index(next_activity)
                #     for follower in all_followers:
                #         next_next, _ = get_forward_star(original, follower)
                #         if next_activity_idx in next_next:
                #             if follower not in in_variant_followers:
                #                 in_variant_followers[follower] = set()
                #             in_variant_followers[follower].add(next_activity_idx)

                next_places = nextt[activity]
                selected_next_places = [np for np in next_places if nextt[np] in activities[ii+1:]]

                # considered_places = set()
                for place in selected_next_places:
                    place_idx = nodes.index(place)
                    next_activities = set(nextt[place])
                    next_activities_idx = [nodes.index(act) for act in next_activities]

                    if len(next_activities) == 0:
                        continue

                    src = activity_idx
                    src_f = embeddings[src].clone()
                    place_f = embeddings[place_idx]
					
                    dst_f = embeddings[next_activities_idx[0]].clone()
                    if len(next_activities_idx) > 1:
                        for l in range(1, len(next_activities_idx)):
                            dst_f += embeddings[list(next_activities_idx)[l]].clone()
                    dst_f /= (len(next_activities) + 1)

                    src_f = src_f.unsqueeze(0)
                    dst_f = dst_f.unsqueeze(0)
                    place_f = place_f.unsqueeze(0)
					
                    embeddings_cat = torch.cat((place_f, src_f, dst_f),1)

                    if place_idx not in in_variant_places_temp_info:
                        in_variant_places_temp_info[place_idx] = []
                    in_variant_places_temp_info[place_idx].append(embeddings_cat)

                    # places_info[place_idx] += embeddings_cat.squeeze() + prev_embedding_summary
					
                    # considered_places.add(place_idx)
					
                # for cp in considered_places:
                #     prev_embedding_summary += places_info[cp]
							
                # embeddings = self.encoder(embeddings, edge_index)
					
                # final_places_info += places_info

            for p,e in in_variant_places_temp_info.items():
                if p not in places_temp_info:
                    places_temp_info[p] = []
                places_temp_info[p].append(torch.mean(torch.stack(e)))

        for p,e in places_temp_info.items():
            final_places_info[p] += torch.mean(torch.stack(e))

        return final_places_info