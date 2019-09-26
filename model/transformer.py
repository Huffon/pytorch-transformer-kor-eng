import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def create_mask(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]

        source_mask = (source != self.params.pad_idx).unsqueeze(1).unsqueeze(2)
        target_pad_mask = (target != self.params.pad_idx).unsqueeze(1).unsqueeze(3)

        # source_mask     = [batch size, 1, 1, source length]
        # target_pad_mask = [batch size, 1, target length, 1]

        target_len = target.shape[1]

        target_sub_mask = torch.tril(torch.ones((target_len, target_len), device=self.params.device)).bool()
        # target_sub_mask = [target length, target length]

        target_mask = target_pad_mask & target_sub_mask
        # target_mask [batch size, 1, target length, target length]

        return source_mask, target_mask

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]

        source_mask, target_mask = self.create_mask(source, target)

        source = self.encoder(source, source_mask)
        output = self.decoder(target, source, target_mask, source_mask)

        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
