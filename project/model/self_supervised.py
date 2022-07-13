import random
import torch

from modules.decoder import DecoderMLP
from modules.encoders import FirstEncoder, SecondEncoder
from utils.graph_utils import get_forward_star, get_next_activities


class SelfSupPredictor(torch.nn.Module):
  def __init__(self, num_node_features, features_size, output_size, encoder, device):
    super(SelfSupPredictor, self).__init__()
    self.output_size = output_size
    self.device = device
    self.first_encoder = FirstEncoder(num_node_features, features_size, kind=encoder)
    self.second_encoder = SecondEncoder(features_size, features_size, kind=encoder)
    self.decoder = DecoderMLP(features_size*3, output_size)
    self.log_sigmoid = torch.nn.LogSigmoid()

  def forward(self, data):
    if self.training:
      return self.forward_training(data)
    else:
      return self.inference(data)

  def forward_training(self, data):
    x, edge_index, original, y, nodes, variants = data

    x = x.to(self.device)
    edge_index = edge_index.to(self.device)

    def get_followers(idx, edge_index):
      followers = set()
      for i in range(len(edge_index[0])):
        if edge_index[0][i].item() == idx:
          followers.add(edge_index[1][i].item())
      return followers

    predictions = torch.zeros(len(nodes), self.output_size, device=self.device)
    features = self.first_encoder(x, edge_index)

    # total = sum([variant['count'] for variant in variants])

    random.shuffle(variants)

    prev_features = features

    for variant in variants:
      features = prev_features
      activities = variant['variant']
      prediction = torch.zeros(len(nodes), self.output_size, device=self.device)
      prev_prob = 0
      for i, activity in enumerate(activities):
        activity_idx = nodes.index(activity)
        all_followers = get_followers(activity_idx, original)

        if len(all_followers) == 0:
          continue

        in_variant_followers = {}
        for j, next_activity in enumerate(activities[i+1:]):
          next_activity_idx = nodes.index(next_activity)
          for follower in all_followers:
            next_next = get_followers(follower, original)
            if next_activity_idx in next_next:
              if follower not in in_variant_followers:
                in_variant_followers[follower] = set()
              in_variant_followers[follower].add(next_activity_idx)
        
        predicted_places = set()

        for place, destination in in_variant_followers.items():
          src = activity_idx

          src_f = features[src].clone()
          place_f = features[place]

          dst_f = features[list(destination)[0]].clone()
          if len(destination) > 1:
            for l in range(1, len(destination)):
              dst_f += features[list(destination)[l]].clone()       

          src_f = src_f.unsqueeze(0)
          dst_f = dst_f.unsqueeze(0)
          place_f = place_f.unsqueeze(0)

          features_cat = torch.cat((place_f, src_f, dst_f),1)

          place_prediction = self.decoder(features_cat)

          place_prediction = self.log_sigmoid(place_prediction)

          prediction[place] += place_prediction.squeeze()

          prev_prob += prediction[place]
          predicted_places.add(place)

        if len(predicted_places) > 0:
          prediction[list(predicted_places)] += prev_prob
         
        features = self.second_encoder(features, edge_index)

      predictions += prediction
    
    return predictions


  def inference(self, data):
    with torch.no_grad():
      prediction = self.forward_training(data)
      _, order = torch.sort(prediction.squeeze(), descending=True)
      order_map = {n.item():i for i,n in enumerate(order)}
  
      result = ['p' not in node for node in data[4]]
      
      activities = [i for i in range(data[4].index('|')+1)]
      connections = {activity:get_next_activities(data[2], activity) for activity in activities}

      checked_places = set()

      keep_going = True
      ok_activities = set()

      while keep_going:
        for activity in activities:
          if activity not in ok_activities:
            discovered_next = get_next_activities(data[2], activity, result)
            if discovered_next != connections[activity]:
              next_places, _ = get_forward_star(data[2], activity)
              next_places.sort(key = lambda x: order_map[x])
              for n in next_places:
                if n not in checked_places:
                  result[n] = True
                  checked_places.add(n)
                  break
            else:
              ok_activities.add(activity)
          
        keep_going = len(ok_activities) < len(connections)

      return result