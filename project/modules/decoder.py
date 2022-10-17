import torch
from torch.nn import Linear, LeakyReLU, Dropout


class DecoderMLP(torch.nn.Module):
  def __init__(self, input_size, output_size):
    super(DecoderMLP, self).__init__()
    self.dense1 = Linear(input_size, input_size*4)
    self.dense2 = Linear(input_size*4, input_size*2)
    self.dense3 = Linear(input_size*2, output_size)
    self.relu = LeakyReLU(0.1)
    self.dropout = Dropout(0.5)
        
  def forward(self, x):   
    output = self.dense1(x)
    output = self.relu(output)
    output = self.dropout(output)
    output = self.dense2(output)
    output = self.relu(output)
    output = self.dropout(output)
    output = self.dense3(output)
    output = self.relu(output)
        
    return output