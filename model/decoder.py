import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_non_pad_mask, create_subsequent_mask, create_target_mask


class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim)

        self.self_attention = MultiHeadAttention(params)
        self.encoder_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, target, encoder_output, target_mask, dec_enc_mask, target_non_pad):
        # target          = [batch size, target length, hidden dim]
        # encoder_output  = [batch size, source length, hidden dim]
        # target_mask     = [batch size, target length, target length]
        # dec_enc_mask    = [batch size, target length, source length]
        # target_non_pad  = [batch size, target length, 1]

        # Apply 'Add & Normalize' self attention, Encoder's Self attention and Position wise Feed Forward Network
        output = self.layer_norm(target + self.self_attention(target, target, target, target_mask))

        # In Decoder stack, query is the output from below layer and key & value are the output from the Encoder
        output = self.layer_norm(output + self.encoder_attention(output, encoder_output, encoder_output, dec_enc_mask))
        output = output * target_non_pad

        output = self.layer_norm(output + self.position_wise_ffn(output))
        output = output * target_non_pad
        # output = [batch size, target length, hidden dim]

        return output


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.device = params.device
        self.hidden_dim = params.hidden_dim

        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx)
        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])
        self.fc = nn.Linear(params.hidden_dim, params.output_dim)

        self.dropout = nn.Dropout(params.dropout)
        self.scale = torch.sqrt(torch.FloatTensor([params.hidden_dim])).to(self.device)

    def forward(self, target, source, encoder_output):
        # target              = [batch size, target length]
        # source              = [batch size, source length]
        # encoder_output      = [batch size, source length, hidden dim]
        subsequent_mask = create_subsequent_mask(target)
        target_mask, dec_enc_mask = create_target_mask(source, target, subsequent_mask)
        # target_mask  = [batch size, target length, target length]
        # dec_enc_mask = [batch size, target length, source length]
        target_non_pad = create_non_pad_mask(target)  # [batch size, target length, 1]

        embedded = self.token_embedding(target)
        positional_encoding = create_positional_encoding(target, self.hidden_dim)
        target = self.dropout(embedded + positional_encoding)

        for decoder_layer in self.decoder_layers:
            target = decoder_layer(target, encoder_output, target_mask, dec_enc_mask, target_non_pad)
        # target = [batch size, target length, hidden dim]

        output = self.fc(target)
        # output = [batch size, target length, output dim]
        return output
