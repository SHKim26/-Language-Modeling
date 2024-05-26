import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset
        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    Args:
        input_file: txt file
    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """
    def __init__(self, input_file, seq_length=30):
        with open(input_file, 'r') as f:
            self.data = f.read()

        self.chars = sorted(list(set(self.data)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.seq_length = seq_length
        self.num_samples = len(self.data) // self.seq_length

        self.char_indices = [self.char_to_idx[ch] for ch in self.data]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = (idx + 1) * self.seq_length

        input_seq = self.char_indices[start_idx:end_idx]
        target_seq = self.char_indices[start_idx+1:end_idx+1]

        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)

        return input_seq, target_seq

if __name__ == '__main__':
    dataset = Shakespeare('shakespeare_train.txt')
    print(f"Dataset length: {len(dataset)}")
    print(f"Sequence length: {dataset.seq_length}")
    print(f"Number of unique characters: {len(dataset.chars)}")

    # Test __getitem__
    input_seq, target_seq = dataset[0]
    print(f"Input sequence shape: {input_seq.shape}")
    print(f"Target sequence shape: {target_seq.shape}")