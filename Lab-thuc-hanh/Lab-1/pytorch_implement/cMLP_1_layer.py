from torch import nn

class cMLP_1_layer(nn.Module):
    """Input → Linear(input_size → 512) → Relu → Linear(512 → 10) → Softmax → Output"""
    """Input → Linear(input_size → 512)  → Softmax → Output là không đúng, vì phải truyền cho softmax input
    là 10 lớp phân loại mong muốn, chứ không phải 512 lớp. Nếu truyền 512 mô hình vẫn học được, nhưng chỉ dùng
    10 neuron đầu tiên để phân loại cho 10 labels, các neuron còn lại bị đánh weight về 0 trong quá trình huấn luyện.
    """
    def __init__(self, input_size, output_size):
        super(cMLP_1_layer, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.softmax(out)
        return out
    
    