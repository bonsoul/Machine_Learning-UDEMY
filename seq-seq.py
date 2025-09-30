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
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    return vocab


english_vocab = build_vocab(english_sentences)
french_vocab = build_vocab(french_sentences)

#tokenize and pad sentences
def tokenize(sentences, vocab, max_len):
    tokenized = []
    for sentence in sentences:
        # convert words to IDs (use <UNK> if word not in vocab)
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()]
        
        # add <SOS> (start) and <EOS> (end) tokens
        tokens = [vocab["<SOS>"]] + tokens + [vocab["<EOS>"]]
        
        # truncate if longer than max_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        # pad if shorter than max_len
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        
        tokenized.append(tokens)
    
    return np.array(tokenized)

max_len_eng = max(len(sentence.split()) for sentence in english_sentences) + 2
max_len_fr = max(len(sentence.split()) for sentence in french_sentences) + 2


english_data = tokenize(english_sentences, english_vocab, max_len_eng)
french_data = tokenize(french_sentences, french_vocab, max_len_fr)


class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])

    
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
    
class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zero(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        input = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden,cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input  =tgt[:, t] if torch.ran(1).item() < teacher_forcing_ratio else top1
        
        return outputs
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = len(english_vocab)
output_dim = len(french_vocab)
embed_dim = 64
hidden_dim = 128
num_layers = 2

encoder = Encoder(input_dim, embed_dim, hidden_dim,num_layers)
decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder, device).to(device)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=french_vocab["<PAD>"])

def train(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            
            
            output = output[:,1:].reshape(-1, output.shape[2])
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1},/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
train(model, dataloader, optimizer, criterion, device)

def translate_sentence(model, sentence, english_vocab, french_vocab, max_len, device):
    model.eval() 
    
    tokens = [english_vocab.get(word, english_vocab["<UNK>"]) for word in sentence.split()]
    tokens = [english_vocab["<SOS>"]] + tokens + [english_vocab["<EOS>"]]
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        hidden, cell = model.encoder(src)
        
    tgt_vocab = {v:k for k, v in french_vocab.items()}
    tgt_indices = [french_vocab["<sos>"]]
    for _ in range(max_len_fr):
        tgt_tensor = torch.tensor([tgt_indices[-1]]).to(device)
        output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)
        pred = output.argmax(1).item()
        tgt_indices.append(pred)
        if pred == french_vocab["<EOS>"]:
            break
    translated_sentence = [tgt_vocab[i] for i in tgt_indices[1:-1]]
    return " ".join(translated_sentence)

#test translation
sentence = "good morning"
translation = translate_sentence(model, sentence, english_vocab, french_vocab, max_len_fr, device)

print(f"Translated Sentence: {translation}")
        
             
        
        