from torch import nn

class cMLP_3_layers(nn.Module):
    """Input → Linear(input_size → 512) → ReLU → Linear(512 → 256) → ReLU → Linear(256 → output_size) → Softmax → Output"""
    def __init__(self, input_size, output_size):
        super(cMLP_3_layers, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # out = self.softmax(out)
        return out