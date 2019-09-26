import torch
import torch.nn as nn

from model.attention import SelfAttention
from model.positionwise import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim)
        self.self_attention = SelfAttention(params)
        self.encoder_attention = SelfAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, target, source, target_mask, source_mask):
        # target         = [batch size, target sentence length, hidden dim]
        # source         = [batch size, source sentence length, hidden dim]
        # target_mask    = [batch size, target sentence length]
        # source_mask    = [batch size, source sentence length]

        target = self.layer_norm(target + self.dropout(self.self_attention(target, target, target, target_mask)))
        target = self.layer_norm(target + self.dropout(self.encoder_attention(target, source, source, source_mask)))
        target = self.layer_norm(target + self.dropout(self.position_wise_ffn(target)))

        return target


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.device = params.device

        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim)
        self.position_embedding = nn.Embedding(1000, params.hidden_dim)

        self.layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])

        self.fc = nn.Linear(params.hidden_dim, params.output_dim)
        self.dropout = nn.Dropout(params.dropout)
        self.scale = torch.sqrt(torch.FloatTensor([params.hidden_dim])).to(self.device)

    def forward(self, target, source, target_mask, source_mask):
        # target         = [batch size, target sentence length]
        # source         = [batch size, source sentence length]
        # target_mask    = [batch size, target sentence length]
        # source_mask    = [batch size, source sentence length]

        position = torch.arange(0, target.shape[1]).unsqueeze(0).repeat(target.shape[0], 1).to(self.device)
        target = self.dropout(self.token_embedding(target) * self.scale) + self.position_embedding(position)

        for layer in self.layers:
            target = layer(target, source, target_mask, source_mask)

        return self.fc(target)
