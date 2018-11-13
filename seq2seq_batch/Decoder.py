import torch
from torch import nn
from attn import Attn

device = 'cuda:0' if torch.cuda.is_available() else "cpu"

# attn :(batch, seq_len)
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, method, num_layers=1):
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
        self.lstm = self.lstm.to(device)
        self.attn = Attn(method, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    # [batch, max_len], hc_state (1, batch, 64) encoder_output: (batch, max_len, 64)
    def forward(self, input, hc_state, encoder_output):
        embedding = self.embedding(input)
        lstm_output, (hidden_state, cell_state) = self.lstm(embedding, hc_state)

        # attn_weight : (batch, max_len)
        attn_weight = self.attn(hidden_state, encoder_output)
        # attn_weight (batch, 1, max_len) encoder_output(batch, max_len, hidden_size)  ==> `context: (batch, 1, hidden_size)
        attn_weight = attn_weight.unsqueeze(1)
        context = attn_weight.bmm(encoder_output)

        # context: (batch, hidden_state)
        context = context.squeeze(1)
        hidden_state = hidden_state.squeeze(0)
        concat_input = torch.cat((context, hidden_state), 1)
        concat_ht = self.concat(concat_input)

        attention_hs = self.tanh(concat_ht)
        output = self.out(attention_hs)
        output = self.softmax(output)
        return output, (hidden_state.unsqueeze(0), cell_state)


# de = Decoder(64, 100, 'general')
# de = de.to(device)
#
# input = torch.LongTensor([[1,2,4,5], [1,2,4,0], [2,2,3,5]]).to(device)
# hidden_state = torch.randn([1,3, 64]).to(device)
# cell_state = torch.randn([1,3, 64]).to(device)
#
# encoder_output = torch.randn([3, 5, 64]).to(device)
# z = de(input, (hidden_state, cell_state), encoder_output)
#
# print(z)
# print(z.size())