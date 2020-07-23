import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isbi):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if isbi:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=isbi)
        self.linear = nn.Linear(hidden_size * self.num_directions + 5, 5)

        if use_cuda:
            self.lstm = self.lstm.cuda()
            self.linear = self.linear.cuda()

    def forward(self, x, hidden, cell, istrain):
        x = x.view(1, len(x), self.input_size)
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = out.squeeze()
        output = torch.cat((out, x.squeeze()), 1)
        output = self.linear(output)
        return output

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)

        if use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()

        return hidden, cell