import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SummaryDataset(Dataset):
    def __init__(self, X, y):
        '''Holds the texts and summaries of the datasets
        Args:
            X: Is a list of tensors with variable length
            y: Is a list of tensors with variable length'''
        self.X = X
        self.y = y
        self.n_samples = len(X)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Summary_collate_fn():
    def __init__(self, pad_id):
        self.pad_id = pad_id 

    def collate_fn(self, batch):
        '''Gets a batch, i.e. a list of tuples (sequence, label) where each of them have different length
        Returns:
            padded_srcs: Batch samples padded to form a tensor of size (B,S)
            src_lengths: tensor of length B containing the original length of all the samples in padded_srcs
            padded_tgts: Batch labels padded to form a tensor of size (B,T)
            tgt_lengths: tensor of length B containing the original length of all the labels in padded_tgt
            '''
        source_seqs, target_seqs = zip(*batch) # two lists each having len == B and variable length of samples

        src_lengths = torch.tensor([len(seq) for seq in source_seqs]) # and max source length is S
        tgt_lengths = torch.tensor([len(seq) for seq in target_seqs]) # and max target length is T

        padded_srcs = pad_sequence(source_seqs, batch_first=True) # Tensor (B,S)
        padded_tgts = pad_sequence(target_seqs, batch_first=True, padding_value=self.pad_id) # Tensor (B,T)

        return padded_srcs, src_lengths, padded_tgts, tgt_lengths
    


def create_packed_dataloader(X, y, pad_id, batch_size, shuffle):
    dataset = SummaryDataset(X,y)

    collate_fn = Summary_collate_fn(pad_id).collate_fn
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,collate_fn=collate_fn, shuffle=shuffle)

    return dataloader
