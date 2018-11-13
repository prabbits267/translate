import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    # batch, max_len
    #  (num_layers * num_directions, batch, hidden_size)
    def forward(self, input, input_len):
        embedded = self.embedding(input)
        packed_input = pack_padded_sequence(embedded, input_len, batch_first=True)
        batch_size = input.size(0)
        seq_len = input.size(1)
        output, (hidden_state, cell_state) = self.lstm(packed_input, None)
        unpacked_output, unpacked_len = pad_packed_sequence(output, batch_first=True)
        unpacked_output = unpacked_output.view(batch_size, seq_len, self.hidden_size)
        return unpacked_output, (hidden_state, cell_state)

    def init_hidden(self):
        hidden_state = Variable(torch.zeros(self.n_layers, 1, self.hidden_size).to(self.device))
        cell_state = Variable(torch.zeros(self.n_layers, 1, self.hidden_size).to(self.device))
        return (hidden_state, cell_state)

# en = Encoder(100, 64, 30)
# en = en.to('cuda:0')
# input = torch.LongTensor([[1,2,4,5], [1,2,4,5], [1,2,0,0]]).to('cuda:0')
# len = torch.LongTensor([4, 4, 2]).to('cuda:0')
# z = en(input, len)
# print(z)