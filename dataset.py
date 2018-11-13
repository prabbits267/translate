from torch.utils.data import Dataset, DataLoader


class Seq2SeqDataset(Dataset):
    def __init__(self):
        self.path = 'data/data.txt'
        self.x_data, self.y_data, self.len, self.full_text, self.vocab = self.read_data()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def read_data(self):
        with open(self.path, 'rt', encoding='utf-8') as file_reader:
            full_text = file_reader.read()
        x_data = list()
        y_data = list()
        for line in full_text.splitlines():
            pair = line.split('\t')
            if(len(pair) == 2):
                x_data.append(pair[0])
                y_data.append(pair[1] + '_')
        return x_data, y_data, len(full_text.splitlines()), full_text, sorted(set(full_text))