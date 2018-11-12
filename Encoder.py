import torch
from torch import nn
from torch.autograd import Variable

device = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_CUDA = torch.cuda.is_available()


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, input):
        embedded = self.embedding(input)
        output, (hidden_state, cell_state) = self.lstm(embedded, self.init_hidden())
        return output, (hidden_state, cell_state)

    def init_hidden(self):
        hidden_state = Variable(torch.zeros(self.n_layers, 1, self.hidden_size).to(self.device))
        cell_state = Variable(torch.zeros(self.n_layers, 1, self.hidden_size).to(self.device))
        return (hidden_state, cell_state)