import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim)

        self.self_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, source, source_mask):
        # source      = [batch size, source length, hidden dim]
        # source_mask = [batch size, source length, source length]

        # Apply 'Add & Normalize' using nn.LayerNorm on self attention and Position wise Feed Forward Network
        output = self.layer_norm(source + self.self_attention(source, source, source, source_mask))
        output = self.layer_norm(output + self.position_wise_ffn(output))
        # output = [batch size, source length, hidden dim]

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.device = params.device

        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.scale = torch.sqrt(torch.FloatTensor([params.hidden_dim])).to(self.device)

    def forward(self, source, source_mask, positional_encoding):
        # source              = [batch size, source length]
        # source mask         = [batch size, source length, source length]
        # positional encoding = [batch size, source length, hidden dim]

        # define positional encoding which encodes token's positional information
        # print(f'[E] Before embedding: {source.shape}')
        embedded = self.token_embedding(source)
        # print(f'[E] After embedding: {embedded.shape}')

        source = self.dropout(embedded + positional_encoding)
        # source = [batch size, source length, hidden dim]

        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask)
        # source = [batch size, source length, hidden dim]
        # print(f'[E] After encoding: {source.shape}')
        # print('------------------------------------------------------------')
        return source
