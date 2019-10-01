import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]
        encoder_output = self.encoder(source)                  # [batch size, source length, hidden dim]
        output = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
