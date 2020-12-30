import torch.nn as nn
import torch.nn.functional as F

class StressModel(nn.Module):
    def __init__(self, input_features=9, output_features=4):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, xb):
        out = self.linear(xb)
        return out
