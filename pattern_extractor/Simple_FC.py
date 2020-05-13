from torch import nn


class SimpleFC(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(SimpleFC, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


