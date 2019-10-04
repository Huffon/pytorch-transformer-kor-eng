import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_non_pad_mask, create_source_mask, create_position_vector


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim)
        self.self_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, source, source_mask, source_non_pad):
        # source          = [batch size, source length, hidden dim]
        # source_mask     = [batch size, source length, source length]
        # source_non_pad  = [batch size, source length, 1]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        normalized_source = self.layer_norm(source)
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)
        output = output * source_non_pad

        normalized_output = self.layer_norm(output)
        output = output + self.position_wise_ffn(normalized_output)
        output = output * source_non_pad
        # output = [batch size, source length, hidden dim]

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.device = params.device
        self.hidden_dim = params.hidden_dim

        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx)
        nn.init.xavier_normal_(self.token_embedding.weight)
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)

        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim)

    def forward(self, source):
        # source = [batch size, source length]
        source_mask = create_source_mask(source)      # [batch size, source length, source length]
        source_non_pad = create_non_pad_mask(source)  # [batch size, source length, 1]
        source_pos = create_position_vector(source)  # [batch size, source length]

        source = self.dropout(self.token_embedding(source) + self.pos_embedding(source_pos))
        # source = [batch size, source length, hidden dim]

        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask, source_non_pad)
        # source = [batch size, source length, hidden dim]
        source = self.layer_norm(source)

        return source
