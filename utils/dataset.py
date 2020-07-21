import torch
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
import numpy as np

import torch.multiprocessing
from multiprocessing import cpu_count

torch.manual_seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')


def read_all_shards(partition, data_dir):
    shards = []
    for fn in os.listdir(os.path.join(data_dir, partition)):
        with open(os.path.join(data_dir, partition, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    return pd.concat(shards)


def prepare_sequence(seq, vocab, padding):
    """
    function to pre-process a single protein sequence
    """
    res = ['<PAD>'] * padding
    res[:min(padding, len(seq))] = seq[:min(padding, len(seq))]
    idxs = [vocab[w] for w in res]
    return idxs


def create_mask(sequence, padding, mask_ratio):
    true_len = len(sequence)
    mask = [1]*padding
    start, end = int(true_len * (1 - mask_ratio)), true_len
    for i in range(start, end):
        mask[i] = 0
    return mask


class PfamPartition(Dataset):
    def __init__(self, data, fams, vocab, padding, fam_vocab, mask_ratio):
        partition = data[data["family_id"].isin(fams)]
        x_raw = partition['aligned_sequence'].values
        y_raw = partition['family_id'].values
        self.x_data = torch.tensor([prepare_sequence(x, vocab, padding) for x in x_raw])
        self.mask = torch.tensor([create_mask(x, padding, mask_ratio) for x in x_raw])
        self.y_data = torch.tensor([fam_vocab[y] for y in y_raw])
        self.length = self.x_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.mask[idx]


class PfamDataset:
    def __init__(self, data_dir='data/random_split', sample_every=1, padding=3000, mask_ratio=0.1):
        self.vocab = {"<PAD>": 0, ".": 1, "M": 2, "H": 3, "B": 4, "W": 5, "R": 6, "U": 7, "I": 8, "O": 9, "L": 10, "T": 11,
                 "D": 12, "F": 13, "X": 14, "Q": 15, "K": 16, "N": 17, "A": 18, "E": 19, "Y": 20, "V": 21, "Z": 22,
                 "S": 23, "P": 24, "C": 25, "G": 26, "<MASK>": 27}

        train = read_all_shards('train', data_dir)
        val = read_all_shards('dev', data_dir)
        test = read_all_shards('test', data_dir)

        fams = np.array(train["family_id"].value_counts().index)[::sample_every]
        fam_vocab = {fam: idx for idx, fam in enumerate(fams)}

        self.train_set = PfamPartition(train, fams, self.vocab, padding, fam_vocab, mask_ratio)
        self.val_set = PfamPartition(val, fams, self.vocab, padding, fam_vocab, mask_ratio)
        self.test_set = PfamPartition(test, fams, self.vocab, padding, fam_vocab, mask_ratio)

        self.target_size = len(fams)

    def get_train_loader(self, batch_size=128, shuffle=True, num_workers=cpu_count()):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_val_loader(self, batch_size=128, shuffle=False, num_workers=cpu_count()):
        return DataLoader(self.val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_test_loader(self, batch_size=128, shuffle=False, num_workers=cpu_count()):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    dataset = PfamDataset(sample_every=10)
    print(dataset.target_size)
    train_loader = dataset.get_train_loader()
    for i, data in enumerate(train_loader, 0):
        inputs, labels, mask = data
        print(inputs.shape, labels.shape, mask.shape)
        if i == 5:
            break
