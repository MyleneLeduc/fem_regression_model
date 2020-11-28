import torch.nn as nn
import torch.nn.functional as F

class StressModel(nn.Module):
    def __init__(self, input_features=9, hidden_layer1=25, hidden_layer2=30, output_features=4, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hidden_layer1)                  
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)                  
        self.out = nn.Linear(hidden_layer2, output_features)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)           

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x

