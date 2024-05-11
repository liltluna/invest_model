import torch
import csv
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Convert each row to a tensor
                sample = torch.tensor([float(value)
                                      for value in row], dtype=torch.float32)
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - \
            len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # (1, seq_len) & (1, seq_len, seq_len),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class GraphDataset(Dataset):

    def __init__(self, ds, seq_len, num_classes):
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        label = int(src_target_pair[0].item())  # scalar
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1
        graph = src_target_pair[2:]  # length 225
        close = src_target_pair[1]  # length 1

        seq_len = graph.size(0)  # len_seq
        encoder_mask = torch.zeros(1, 1, seq_len, dtype=torch.bool)

        encoder_input = torch.cat(
            [
                torch.tensor(graph, dtype=torch.float),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                label_tensor
            ],
            dim=0,
        )

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "encoder_mask": encoder_mask,
            "label": label,  # (seq_len)
            "close": close
        }


class BasicGraphDataset(Dataset):
    def __init__(self, file_path, type:str = 'train'):
        self.data = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Convert each row to a tensor
                date = row[0]
                data_daily = torch.tensor([float(value)
                                      for value in row[1:]], dtype=torch.float32)
                self.data.append([date, data_daily])
        if type == 'train':
            print('train data will be duplicated')
            l0_list = [x for x in self.data if x[1][0] == 0]
            l1_list = [x for x in self.data if x[1][0] == 1]
            l2_list = [x for x in self.data if x[1][0] == 2]
            l1_new = []
            l2_new = []
            l1_ratio = len(l0_list) // len(l1_list)
            l2_ratio = len(l0_list) // len(l2_list)
            for row in self.data:
                if row[1][0] == 1:
                    for _ in range(l1_ratio):
                        l1_new.append(row)
                if row[1][0] == 2:
                    for _ in range(l2_ratio):
                        l2_new.append(row)
            self.data = self.data + l1_new + l2_new
        print(f'all data length:{len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GraphDataset(Dataset):

    def __init__(self, ds, seq_len, num_classes):
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ds) - 1

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        label_target_pair = self.ds[idx + 1]
        date = label_target_pair[0]
        label = label_target_pair[1][0]
        graph = src_target_pair[1][2:]  # length 225
        close = src_target_pair[1][1]  # length 1

        nn_input = torch.cat(
            [
                torch.tensor(graph, dtype=torch.float),
            ],
            dim=0,
        )

        label = torch.tensor(label, dtype=torch.int)

        return {
            "date": date, 
            "nn_input": nn_input, 
            "label": label,
            "close": close
        }