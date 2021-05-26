
from torch import nn


class Autoencoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.e1 = nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["latent_dim"])
        self.e1_act = nn.CELU()
        self.e2 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["latent_dim"])
        self.e2_act = nn.Identity()
        self.d1 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["latent_dim"])
        self.d1_act = nn.CELU()
        self.d2 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["input_shape"])
        self.d2_act = nn.Identity()

    def encode(self, features):
        encoded = self.e1(features)
        encoded = self.e1_act(encoded)
        encoded = self.e2(encoded)
        encoded = self.e2_act(encoded)
        return encoded

    def decode(self, encoded):
        decoded = self.d1(encoded)
        decoded = self.d1_act(decoded)
        decoded = self.d2(decoded)
        decoded = self.d2_act(decoded)
        return decoded

    def forward(self, features):
        encoded = self.encode(features)
        decoded = self.decode(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)