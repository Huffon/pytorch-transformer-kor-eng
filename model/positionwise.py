import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(params.hidden_dim, params.pf_dim, 1)
        self.conv2 = nn.Conv2d(params.pf_dim, params.hidden_dim, 1)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        # x = [batch size, source(or target) length, hidden dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hidden dim, source(or target) length]

        x = self.dropout(F.relu(self.conv1(x)))
        # x = [batch size, pf dim, source(or target) length)

        x = self.dropout(F.relu(self.conv2(x)))
        # x = [batch size, hidden dim, source(or target) length)

        x = x.permute(0, 2, 1)
        # x = [batch size, source(or target) length, hidden dim]

        return x
