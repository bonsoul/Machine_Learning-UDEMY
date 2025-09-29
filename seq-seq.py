# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


#example English to French entences

english_sentences = ["hello", "how are you", "good morning", "thank you","good night"]
french_sentences = ["bonjour","comment ca va","bon matin","merci","bonne nuit"]


#vocanulary and tokenization
def build_vocab(sentences):
    vocab = {"<PAD>":0, "<505>": 1, "<EOS>":2, "<UNK>":3}
    for sentence in sentences:
        for word in sentences.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

english_vocab = build_vocab(english_sentences)
french_vocab = build_vocab(french_sentences)

#tokenize and pad sentences
def tokenize(sentences, vocab, max_len):
    tokenized = []
    for sentence in sentences:
        tokens = [vocab.get(word, vocab['<UNK>']) for word in sentence.split()]
        tokens = [vocab["<SOS>"]] + tokens + [vocab["<EOS>"]]
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        tokenized.append(tokens)
    return np.array(tokenized)

max_len_eng = max(len(sentence.split()) for sentence in english_sentences) + 2
max_len_fr = max(len(sentence.split()) for sentence in french_sentences) + 2


english_data = tokenize(english_sentences, english_vocab, max_len_eng)
french_data = tokenize(french_sentences, french_vocab, max_len_fr)


class TranslationDataset(Dataset):
    def __init__(self,src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data(idx))
    
    
dataset = TranslationDataset(english_data, french_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell
        