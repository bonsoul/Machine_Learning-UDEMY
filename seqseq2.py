# corrected seq2seq example

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# example English to French sentences
english_sentences = ["hello", "how are you", "good morning", "thank you", "good night"]
french_sentences  = ["bonjour", "comment ca va", "bon matin", "merci", "bonne nuit"]

# vocabulary builder
def build_vocab(sentences):
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

english_vocab = build_vocab(english_sentences)
french_vocab  = build_vocab(french_sentences)

# tokenize and pad sentences
def tokenize(sentences, vocab, max_len):
    tokenized = []
    for sentence in sentences:
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()]
        tokens = [vocab["<SOS>"]] + tokens + [vocab["<EOS>"]]
        # truncate if longer than max_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        # pad if shorter than max_len
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        tokenized.append(tokens)
    return np.array(tokenized, dtype=np.int64)

max_len_eng = max(len(s.split()) for s in english_sentences) + 2  # +SOS +EOS
max_len_fr  = max(len(s.split()) for s in french_sentences) + 2

english_data = tokenize(english_sentences, english_vocab, max_len_eng)
french_data  = tokenize(french_sentences, french_vocab, max_len_fr)

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # ensure long dtype for embedding indices
        return torch.tensor(self.src_data[idx], dtype=torch.long), torch.tensor(self.tgt_data[idx], dtype=torch.long)

dataset = TranslationDataset(english_data, french_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)                 # (batch, seq_len, emb)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        # x: (batch,) token ids -> convert to (batch, 1)
        x = x.unsqueeze(1)
        embedded = self.embedding(x)                  # (batch, 1, emb)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))     # (batch, output_dim)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features

        # initialize outputs (float) on correct device
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=self.device)

        hidden, cell = self.encoder(src)

        input = tgt[:, 0]  # first token (<SOS>) for each batch
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)  # output: (batch, vocab)
            outputs[:, t, :] = output
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio
            input = tgt[:, t] if teacher_force else top1

        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = len(english_vocab)
output_dim = len(french_vocab)
embed_dim = 64
hidden_dim = 128
num_layers = 2

encoder = Encoder(input_dim, embed_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=french_vocab["<PAD>"])

def train(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt)

            # shift out first token (<SOS>) when computing loss
            output = output[:, 1:, :].reshape(-1, output.shape[2])  # (batch*(tgt_len-1), vocab)
            tgt_flat = tgt[:, 1:].reshape(-1)                       # (batch*(tgt_len-1))

            loss = criterion(output, tgt_flat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

train(model, dataloader, optimizer, criterion, device, num_epochs=50)

def translate_sentence(model, sentence, english_vocab, french_vocab, max_len_eng, max_len_fr, device):
    model.eval()

    tokens = [english_vocab.get(word, english_vocab["<UNK>"]) for word in sentence.split()]
    tokens = [english_vocab["<SOS>"]] + tokens + [english_vocab["<EOS>"]]
    # pad to expected encoder length
    if len(tokens) < max_len_eng:
        tokens += [english_vocab["<PAD>"]] * (max_len_eng - len(tokens))
    else:
        tokens = tokens[:max_len_eng]

    src = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

    with torch.no_grad():
        hidden, cell = model.encoder(src)

    # reverse mapping for french vocab
    id2word = {idx: word for word, idx in french_vocab.items()}

    # start token for decoder
    tgt_indices = [french_vocab["<SOS>"]]

    for _ in range(max_len_fr):
        tgt_tensor = torch.tensor([tgt_indices[-1]], dtype=torch.long).to(device)
        output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)  # (1, vocab)
        pred = output.argmax(1).item()
        tgt_indices.append(pred)
        if pred == french_vocab["<EOS>"]:
            break

    # map indices to words, skip start and end tokens
    translated_sentence = [id2word.get(i, "<UNK>") for i in tgt_indices[1:] if i != french_vocab["<EOS>"]]
    return " ".join(translated_sentence)

# test translation (after training)
sentence = "good morning"
translation = translate_sentence(model, sentence, english_vocab, french_vocab, max_len_eng, max_len_fr, device)
print(f"Translated Sentence: {translation}")
