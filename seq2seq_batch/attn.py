import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()

# input : decoder_output, current hidden_state
# output : attn energy (batch, seq_len)
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    # single sentence ht, hs
    # hidden (num_layers * num_directions, batch, hidden_size) (1, 32, 64)
    # encoder output (batch, time_step, hidden_size)            (32, max_timestep, 64) ==> batch, 64
    # seq_len: array of max time step [<batch>]
    def forward(self, hidden, encoder_output):
        hidden = hidden.squeeze(0)
        batch_size = encoder_output.size(0)
        max_seq_len = encoder_output.size(1)
        attn_energies = Variable(torch.zeros(batch_size, max_seq_len))
        if USE_CUDA:
            attn_energies = attn_energies.cuda()
        # for i in range(seq_len):
        #     attn_energies[i] = self.score(hidden, encoder_output[i])
        # hidden[i]: [64],  encoder_output[i]:(max_timestep, 64)
        # return (batch, max_len)
        for i in range(batch_size):
            attn_energies[i] = self.score(hidden[i], encoder_output[i])
        return F.softmax(attn_energies, dim=0)

    # hidden (64)
    # encoder_output: (max_time_step, 64)
    # return max_time_step
    def score(self, hidden, encoder_output):
        seq_len = encoder_output.size(0)
        energy = torch.zeros([seq_len])
        if self.method == 'dot':
            for i in range(seq_len):
                energy[i] = torch.dot(hidden.view(-1), encoder_output[i].view(-1))
            return energy

        elif self.method == 'general':
            for i in range(seq_len):
                z = encoder_output[i]
                energy_attn = self.attn(encoder_output[i])
                energy[i] = torch.dot(hidden, energy_attn)
                return energy
        else:
            for i in range(seq_len):
                energy_attn = self.attn(torch.cat((hidden[0], encoder_output[i]), 0))
                energy = self.v.dot(energy_attn)
                return energy




# attn = Attn('general', 64)
# hidden = torch.randn(1, 32, 64)
# encoder_output = torch.randn(32, 15, 64)
# a = attn(hidden, encoder_output)
# print(a)
# print(a.size())


