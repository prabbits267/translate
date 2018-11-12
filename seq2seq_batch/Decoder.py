import torch
from torch import nn
from attn import Attn

device = 'cuda:0' if torch.cuda.is_available() else "cpu"

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, method ,num_layers=1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.method = method

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size = hidden_size,
            hidden_size = hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.attn = Attn(method, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    # input ([[1]]) hc_state ((1,1,64), (1,1,64)) encoder_output (1,<time_step>,64)
    def forward(self, input, hc_state, encoder_output):
        embedding = self.embedding(input)
        lstm_output, (hidden_state, cell_state) = self.lstm(embedding, hc_state)

        # (1, 1, 64) , encoder_output (1, <time_step>, 64) ==> (<time_step>)
        attn_weight = self.attn(hidden_state, encoder_output)
        # attn_weight (1, time_step) encoder_output(time_step, hidden_size)  ==> context: (1, hidden_size)
        attn_weight = attn_weight.unsqueeze(0)
        encoder_output = encoder_output.squeeze(0)
        context = attn_weight.mm(encoder_output)

        # context: (1, hidden_size)  hidden_state (1, 64) ==> (1, 64)
        hidden_state = hidden_state.squeeze(0)
        concat_input = torch.cat((context, hidden_state), 1)
        concat_ht = self.concat(concat_input)

        attention_hs = self.tanh(concat_ht)

        output = self.out(attention_hs)
        output = self.softmax(output)

        return output, (hidden_state.unsqueeze(0), cell_state)