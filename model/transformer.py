import numpy as np
import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.params = params
        self.hidden_dim = params.hidden_dim

        self.device = params.device
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def create_subsequent_mask(self, target):
        # target = [batch size, target length]

        batch_size, target_length = target.size()
        '''
        if target length is 5 and diagonal is 1, this function returns
            [[0, 1, 1, 1, 1],
             [0, 0, 1, 1, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1]]
        '''
        # torch.triu returns the upper triangular part of a matrix based on user defined diagonal
        subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(self.device)
        # subsequent_mask = [target length, target length]

        # repeat subsequent mask 'batch size' times to cover all data instances in the batch
        subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        # subsequent_mask = [batch size, target length, target length]

        return subsequent_mask

    def create_mask(self, source, target, subsequent_mask):
        # source          = [batch size, source length]
        # target          = [batch size, target length]
        # subsequent_mask = [batch size, target length, target length]
        source_length = source.shape[1]
        target_length = target.shape[1]

        # create boolean tensors which will be used to mask padding tokens of both source and target sentence
        source_mask = (source == self.params.pad_idx)
        target_mask = (target == self.params.pad_idx)
        # source_mask    = [batch size, source length]
        # target_mask    = [batch size, target length]
        '''
        if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
        masking tensor will be [False, False, False, False, False, True, True, True, False]
        '''
        # repeat sentence masking tensors 'sentence length' times
        dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
        source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)
        target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

        # source_mask    = [batch size, source length, source length]
        # target_mask    = [batch size, target length, target length]
        # dec_enc_mask   = [batch size, target length, source length]

        # combine <pad> token masking tensor and subsequent masking tensor for decoder's self attention
        target_mask = target_mask | subsequent_mask
        # target_mask = [batch size, target length, target length]

        return source_mask, target_mask, dec_enc_mask

    def create_non_pad_mask(self, sentence):
        # padding token shouldn't be used for the output tensor
        # to use only non padding token, create non-pad masking tensor
        return sentence.ne(self.params.pad_idx).type(torch.float).unsqueeze(-1)

    def create_positional_encoding(self, batch_size, sentence_len):
        # PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim)
        # PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim)
        sinusoid_table = np.array([pos/np.power(10000, 2*i/self.hidden_dim)
                                   for pos in range(sentence_len) for i in range(self.hidden_dim)])
        # sinusoid_table = [sentence length * hidden dim]

        sinusoid_table = sinusoid_table.reshape(sentence_len, -1)
        # sinusoid_table = [sentence length, hidden dim]

        sinusoid_table[0::2, :] = np.sin(sinusoid_table[0::2, :])  # calculate pe for even numbers
        sinusoid_table[1::2, :] = np.sin(sinusoid_table[1::2, :])  # calculate pe for odd numbers

        # convert numpy based sinusoid table to torch.tensor and repeat it 'batch size' times
        sinusoid_table = torch.FloatTensor(sinusoid_table).to(self.device)
        sinusoid_table = sinusoid_table.unsqueeze(0).repeat(batch_size, 1, 1)
        # sinusoid_table = [batch size, sentence length, hidden dim]

        return sinusoid_table

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]
        source_batch, source_len = source.size()
        target_batch, target_len = target.size()

        # create masking tensor for self attention (encoder & decoder) and decoder's attention on the output of encoder
        subsequent_mask = self.create_subsequent_mask(target)
        source_mask, target_mask, dec_enc_mask = self.create_mask(source, target, subsequent_mask)

        # create non-pad masking tensor which will be used to extract non-padded tokens from output
        source_non_pad = self.create_non_pad_mask(source)
        target_non_pad = self.create_non_pad_mask(target)
        # non_pad = [batch size, sentence length, 1]

        source_positional_encoding = self.create_positional_encoding(source_batch, source_len)
        target_positional_encoding = self.create_positional_encoding(target_batch, target_len)

        source = self.encoder(source, source_mask, source_positional_encoding, source_non_pad)
        output = self.decoder(target, source, target_mask, dec_enc_mask, target_positional_encoding, target_non_pad)
        # output = [batch size, target length, output dim]

        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
