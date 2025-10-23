from torch import nn

class cMLP_1_layer(nn.Module):
    """Input → Linear(input_size → 10) → Softmax → Output"""
    def __init__(self, input_size, output_size):
        super(cMLP_1_layer, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.softmax(out)
        return out
    
    