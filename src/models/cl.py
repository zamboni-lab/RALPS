
from torch import nn


class Classifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.l1 = nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["n_batches"])
        self.l1_act = nn.LeakyReLU()
        self.l2 = nn.Linear(in_features=kwargs["n_batches"], out_features=kwargs["n_batches"])
        self.l2_act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l1_act(x)
        x = self.l2(x)
        x = self.l2_act(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)