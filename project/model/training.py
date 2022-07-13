import numpy as np
import torch


class Trainer():
  def __init__(self, model, optimizer, device):
    self.model = model
    self.optimizer = optimizer
    self.device = device

  def train(self, data, max_runs):
    nodes = data[4]

    cumulative_loss = []
    
    prev_prediction = torch.zeros(len(nodes), 1, device=self.device)
    difference = lambda x, y: abs((-torch.sum(x)-torch.sum(y)).item())
    prev_difference = float('+inf')
      
    self.model.train()
    self.optimizer.zero_grad()
    prediction = self.model(data)
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
      prediction = self.model(data)
      loss = -torch.sum(prediction)
      loss.backward()
      self.optimizer.step()
      current_difference = difference(prediction, prev_prediction)
      max_runs -= 1

    return self.model, np.mean(cumulative_loss)

  def test(self, data):
    self.model.eval()
    return self.model(data) 