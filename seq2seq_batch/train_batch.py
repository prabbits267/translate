import torch
from torch.nn import NLLLoss
from torch.utils.data import DataLoader

from Decoder import Decoder
from Encoder import Encoder
from attn import Attn
from dataset import Seq2SeqDataset


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
        seq_tensor = torch.zeros([batch_size, seq_len]).to(self.device)
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
        return input_seq, input_len, target_seq

    def data_index(self):
        char2index = {}
        char2index['_'] = 0
        char2index.update({w:i+1 for i, w in enumerate(self.vocab)})
        index2char = {w[1]:w[0] for w in char2index.items()}
        return char2index, index2char

    # on sigle batch ()
    def step(self, input, target):
        input_seq, input_len, target_seq = self.create_batch(input, target)
        encoder_output, (hidden_state, cell_state) = self.encoder(input_seq, input_len)
        return encoder_output, (hidden_state, cell_state)


train = TrainBatch(100, 64, 5, 0.01, 'general')
for i, (input, target) in enumerate(train.data_loader):
    a,b,c = train.step(input, target)
    print('')




