import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.device = params.device
        self.hidden_dim = params.hidden_dim
        self.n_head = params.n_head

        assert params.hidden_dim % params.n_head == 0

        self.query_w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_w = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(params.dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_dim // self.n_head])).to(self.device)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, source(or target) length, hidden dim]
        batch_size = query.shape[0]

        q = self.query_w(query)
        k = self.key_w(key)
        v = self.value_w(value)
        # q, k, v = [batch size, source(or target) length, hidden dim]

        q = q.view(batch_size, -1, self.n_head, self.hidden_dim // self.n_head).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_head, self.hidden_dim // self.n_head).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_head, self.hidden_dim // self.n_head).permute(0, 2, 1, 3)
        # q, k, v = [batch size, num head, source(or target) length, hidden dim // num head]

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, num head, source(or target) length, source(or target) length]

        if mask is not None:
            pass

        attention = self.dropout(F.softmax(energy, dim=-1))
        # attention = [batch size, num head, source(or target) length, source(or target) length]

        x = torch.matmul(attention, v)
        # x = [batch size, num head, source(or target) length, hidden dim // num head]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, source(or target) length, num head, hidden dim // num head]

        x = x.view(batch_size, -1, self.n_head * (self.hidden_dim // self.n_head))
        # x = [batch size, source(or target) length, hidden dim]

        x = self.fc(x)
        # x = [batch size, source(or target) length, hidden dim]

        return x
