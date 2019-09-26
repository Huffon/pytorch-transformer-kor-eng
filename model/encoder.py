import torch
import torch.nn as nn

from model.attention import SelfAttention
from model.positionwise import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim)
        self.self_attention = SelfAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, source, source_mask):
        # source      = [batch size, source length, hidden dim]
        # source_mask = [batch size, source length]

        source = self.layer_norm(source + self.dropout(self.self_attention(source, source, source, source_mask)))
        source = self.layer_norm(source + self.dropout(self.position_wise_ffn(source)))

        return source


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.device = params.device

        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim)
        self.position_embedding = nn.Embedding(1000, params.hidden_dim)

        self.layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])

        self.dropout = nn.Dropout(params.dropout)
        self.scale = torch.sqrt(torch.FloatTensor([params.hidden_dim])).to(self.device)

    def forward(self, source, source_mask):
        # source      = [batch size, source length]
        # source_mask = [batch size, source length]

        # define positional encoding which encodes token's positional information
        position = torch.arange(0, source.shape[1]).unsqueeze(0).repeat(source.shape[0], 1).to(self.device)
        source = self.dropout(self.token_embedding(source) * self.scale) + self.position_embedding(position)

        # source = [batch size, source length, hidden dim]

        for layer in self.layers:
            source = layer(source, source_mask)

        return source
