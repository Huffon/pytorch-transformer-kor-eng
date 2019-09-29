import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.n_head == 0

        self.device = params.device
        self.hidden_dim = params.hidden_dim
        self.n_head = params.n_head

        self.attentions = nn.ModuleList([SelfAttention(params) for _ in self.n_head])

        self.o_w = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, query, key, value):
        # query, key, value = [batch size, sentence length, hidden dim]

        weighted_v = [attention(query, key, value) for attention in self.attentions]
        # weighted_v = [batch size, sentence length, attention dim] * num head

        weighted_v = torch.cat(weighted_v, dim=2)
        # weighted_v = [batch size, sentence length, hidden dim]

        output = nn.Linear(weighted_v)
        # output = [batch size, sentence length, hidden dim]

        return output


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()

        self.device = params.device
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // self.n_head

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim)

        self.dropout = nn.Dropout(params.dropout)

        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(self.device)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q, k.permute(0, 2, 1))
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None:
            pass

        # normalize attention score using soft max function based column
        attention_score = self.dropout(F.softmax(self_attention, dim=2))
        # attention_score = [batch size, sentence length, sentence length]

        # compute weighted value matrix using attention score
        weighted_v = torch.bmm(attention_score, v)
        # weighted_v = [batch size, sentence length, attention dim]

        return weighted_v
