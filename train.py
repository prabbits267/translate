import torch
from torch import nn, optim
from torch.nn import NLLLoss
from torch.utils.data import DataLoader

from Decoder import Decoder
from Encoder import Encoder
from dataset import Seq2SeqDataset

SOS = '_'

class Train():
    def __init__(self, input_size, hidden_size, batch_size, learning_rate, num_epoch, method):
        dataset = Seq2SeqDataset()

        self.vocab = sorted(set(dataset.full_text))
        self.vocab_size = len(self.vocab)
        self.char2ind, self.ind2char = self.get_vocab()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = self.vocab_size
        self.method = method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.encoder = Encoder(input_size, hidden_size, self.vocab_size)
        self.decoder = Decoder(hidden_size, self.output_size, method)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.loss_function = NLLLoss()

        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)



    def step(self, input, output):
        input_tensor = self.convert2indx(input)
        target_tensor = self.convert2indx(output).squeeze(0)

        encoder_output, (hidden_state, cell_state) = self.encoder(input_tensor)

        target_len = target_tensor.size(0)
        SOS_tensor = self.convert2indx(SOS)

        decoder_input = SOS_tensor
        decoder_output = torch.zeros([target_len, self.output_size]).to(self.device)
        output_index = torch.zeros(target_len)
        # use teacher forcing
        for i in range(target_len):
            output, (hidden_state, cell_state) = self.decoder(decoder_input, (hidden_state, cell_state), encoder_output)
            decoder_output[i] = output
            output_index[i] = output.topk(1)[1]
            decoder_input = target_tensor[i].unsqueeze(0).unsqueeze(0)

        loss = self.loss_function(decoder_output, target_tensor)
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        loss.backward()

        self.encoder_optim.step()
        self.decoder_optim.step()

        return loss.data[0], decoder_output, output_index

    def train_batch(self):
        total_loss = 0
        for i, (x_data, y_data) in enumerate(self.dataloader):
            loss, _, output = self.step(x_data[0], y_data[0])
            total_loss += loss
        return total_loss

    def train(self):
        for i in range(self.num_epoch):
            loss = self.train_batch()
            print('Epoch : ', i, ' -->>>--> loss', loss)
            print('output ', self.step('We lost.', 'Nous fûmes défaites.')[2])
            print('output ', self.convert2indx('Nous fûmes défaites.') )

    def convert2indx(self, input):
        input_tensor = torch.LongTensor([[self.char2ind[w] for w in list(input)]])
        return input_tensor.to(self.device)

    def get_vocab(self):
        char2ind = {'_':1}
        char2ind.update({w:i+1 for i, w in enumerate(self.vocab)})
        ind2char = {w[1]:w[0] for w in char2ind.items()}
        return char2ind, ind2char

# input_size, hidden_size, batch_size, learning_rate, method
train = Train(300, 64, 1, 0.01, 50, 'general')
train.train()
