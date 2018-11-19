import torch
from torch.autograd import Variable
from torch.nn import NLLLoss
from torch.utils.data import DataLoader

from Decoder import Decoder
from Encoder import Encoder
from attn import Attn
from dataset import Seq2SeqDataset

SOS = '_'

class TrainBatch():
    def __init__(self, input_size, hidden_size, batch_size, learning_rate, method, num_layers=1):
        dataset = Seq2SeqDataset()
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        self.vocab = dataset.vocab
        self.output_size = len(self.vocab)
        self.char2index, self.index2char = self.data_index()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_layers = 1
        self.method = method

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.attn = Attn(method, hidden_size)
        self.encoder = Encoder(input_size, hidden_size, self.output_size, self.num_layers)
        self.decoder = Decoder(hidden_size, self.output_size, method, self.num_layers)

        self.attn = self.attn.to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.loss_function = NLLLoss()
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

    def word_to_index(self, word):
        char_index = [self.char2index[w] for w in list(word)]
        return torch.LongTensor(char_index).to(self.device)

    # return batch_indedx _ after softed
    def create_batch_tensor(self, batch_word, batch_len):
        batch_size = len(batch_word)
        seq_len = max(batch_len)
        seq_tensor = torch.zeros([batch_size, seq_len]).long().to(self.device)
        for i in range(batch_size):
            seq_tensor[i, :batch_len[i]] = self.word_to_index(batch_word[i])
        return seq_tensor

    def create_batch(self, input, target):
        input_seq = [list(w) for w in list(input)]
        target_seq = [list(w) for w in list(target)]

        seq_pairs = sorted(zip(input_seq, target_seq), key=lambda p: len(p[0]), reverse=True)
        input_seq, target_seq = zip(*seq_pairs)
        input_len = [len(w) for w in input_seq]
        target_len = [len(w) for w in target_seq]

        input_seq = self.create_batch_tensor(input_seq, input_len)
        input_len = torch.LongTensor(input_len).to(self.device)
        target_seq = self.create_batch_tensor(target_seq, target_len)
        return Variable(input_seq), Variable(input_len), Variable(target_seq)

    def data_index(self):
        char2index = {}
        char2index.update({w:i for i, w in enumerate(self.vocab)})
        index2char = {w[1]:w[0] for w in char2index.items()}
        return char2index, index2char

    # on sigle batch ()
    # encoder_output: (batch, max_len, hidden) (5,8,64)
    #
    def step(self, input, target):
        input_seq, input_len, target_seq = self.create_batch(input, target)
        # encoder_output: (batch, max_len, hidden) (5,8,64)
        # hidden (1, batch, 64)
        encoder_output, (hidden_state, cell_state) = self.encoder(input_seq, input_len)
        # SOS_index = torch.LongTensor(self.char2index[SOS]).to(self.device)
        # run one by one
        batch_size = input_seq.size(0)
        max_len = target_seq.size(1)
        decoder_output = torch.zeros([batch_size, max_len, self.output_size]).to(self.device)
        # start of sentence
        decoder_input = torch.tensor((), dtype=torch.long)
        decoder_input = decoder_input.new_ones([batch_size, 1]).to(self.device)
        decoder_input = decoder_input * self.char2index['_']
        input_feed = torch.zeros([batch_size, max_len])

        for i in range(max_len):
            output, (hidden_state, cell_state) = self.decoder(decoder_input, (hidden_state, cell_state), encoder_output)
            input_feed[:, i] = decoder_input.squeeze(1)
            decoder_input = target_seq[:, i].unsqueeze(1)
            decoder_output[:, i] = output
        decoder_output = decoder_output.view(-1, self.output_size)
        target_seq = target_seq.view(-1)

        loss = self.loss_function(decoder_output, target_seq)

        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        loss.backward()

        self.encoder_optim.step()
        self.decoder_optim.step()
        return loss.data[0]

train = TrainBatch(100, 64, 5, 0.01, 'general')
for i in range(10):
    print('================== ', i)
    print()
    for i, (input, target) in enumerate(train.data_loader):
        loss = train.step(input, target)
        print(loss)




