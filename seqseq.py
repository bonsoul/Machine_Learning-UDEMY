# seq2seq_attention.py
import random
import math
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1
BATCH_SIZE = 32
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
MAX_DECODING_LEN = 50
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

# -----------------------
# Toy dataset (character-level: reverse string)
# Replace this with your own dataset loader
# -----------------------
def make_toy_pairs(n=5000, min_len=3, max_len=12):
    """Create toy dataset: input string -> reversed string (character-level)."""
    import string
    chars = string.ascii_lowercase + " "
    pairs = []
    for _ in range(n):
        L = random.randint(min_len, max_len)
        s = "".join(random.choice(chars) for _ in range(L)).strip()
        if not s:
            s = "a"
        src = s
        trg = s[::-1]
        pairs.append((src, trg))
    return pairs

# -----------------------
# Vocab + tokenizer (char-level)
# -----------------------
class Vocab:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []
        self.add_token(PAD_TOKEN)
        self.add_token(SOS_TOKEN)
        self.add_token(EOS_TOKEN)
        self.add_token(UNK_TOKEN)

    def add_token(self, tok: str):
        if tok not in self.token2idx:
            self.token2idx[tok] = len(self.idx2token)
            self.idx2token.append(tok)

    def build_from_pairs(self, pairs: List[Tuple[str, str]]):
        for src, trg in pairs:
            for ch in src:
                self.add_token(ch)
            for ch in trg:
                self.add_token(ch)

    def encode(self, s: str) -> List[int]:
        return [self.token2idx.get(ch, self.token2idx[UNK_TOKEN]) for ch in s]

    def decode(self, idxs: List[int]) -> str:
        tokens = []
        for i in idxs:
            tok = self.idx2token[i]
            if tok == EOS_TOKEN:
                break
            if tok in (PAD_TOKEN, SOS_TOKEN):
                continue
            tokens.append(tok)
        return "".join(tokens)

    def __len__(self):
        return len(self.idx2token)

# -----------------------
# Dataset and collate
# -----------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], vocab: Vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        src_ids = [self.vocab.token2idx[SOS_TOKEN]] + self.vocab.encode(src) + [self.vocab.token2idx[EOS_TOKEN]]
        trg_ids = [self.vocab.token2idx[SOS_TOKEN]] + self.vocab.encode(trg) + [self.vocab.token2idx[EOS_TOKEN]]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)

def collate_fn(batch):
    srcs, trgs = zip(*batch)
    src_lens = [len(x) for x in srcs]
    trg_lens = [len(x) for x in trgs]
    src_pad = nn.utils.rnn.pad_sequence(srcs, padding_value=0, batch_first=True)
    trg_pad = nn.utils.rnn.pad_sequence(trgs, padding_value=0, batch_first=True)
    return src_pad, torch.tensor(src_lens), trg_pad, torch.tensor(trg_lens)

# -----------------------
# Model: Encoder
# -----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, src, src_len):
        # src: (batch, seq)
        embedded = self.embedding(src)  # (batch, seq, emb)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (batch, seq, hidden*2)
        # combine bidirectional hidden states
        # h_n: (num_layers*2, batch, hidden)
        # We'll concat the forward/backward to form initial decoder state
        def _cat_directions(h):
            # h shape: (num_layers*2, batch, hidden) -> (num_layers, batch, hidden*2)
            layers = []
            for i in range(0, h.size(0), 2):
                layers.append(torch.cat([h[i], h[i+1]], dim=1))
            return torch.stack(layers, dim=0)
        h_n_cat = _cat_directions(h_n)
        c_n_cat = _cat_directions(c_n)
        # out: (batch, seq, hidden*2)
        return out, (h_n_cat, c_n_cat)

# -----------------------
# Attention
# -----------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        # enc_hidden_dim is hidden*2 (bidirectional)
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: (batch, dec_hidden)
        # encoder_outputs: (batch, src_len, enc_hidden_dim)
        batch_size, src_len, _ = encoder_outputs.size()
        dec_hidden_exp = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, dec_hidden)
        energy = torch.tanh(self.attn(torch.cat((dec_hidden_exp, encoder_outputs), dim=2)))  # (batch, src_len, dec_hidden)
        scores = self.v(energy).squeeze(2)  # (batch, src_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_hidden_dim)
        return context, attn_weights

# -----------------------
# Decoder
# -----------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_dim, dec_hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim)
        self.rnn = nn.LSTM(enc_hidden_dim + embed_size, dec_hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(enc_hidden_dim + dec_hidden_dim + embed_size, vocab_size)

    def forward(self, input_tok, prev_hidden, prev_cell, encoder_outputs, mask):
        # input_tok: (batch,) token indices (current step)
        embedded = self.embedding(input_tok).unsqueeze(1)  # (batch, 1, emb)
        # prev_hidden: (num_layers, batch, dec_hidden) -> take top layer for attention
        last_hidden = prev_hidden[-1]  # (batch, dec_hidden)
        context, attn_weights = self.attention(last_hidden, encoder_outputs, mask)  # context: (batch, enc_hidden_dim)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch, 1, emb+enc_hidden)
        output, (h_n, c_n) = self.rnn(rnn_input, (prev_hidden, prev_cell))  # output: (batch,1,dec_hidden)
        output = output.squeeze(1)  # (batch, dec_hidden)
        # concat output, context, embedded for final prediction
        pred_input = torch.cat([output, context, embedded.squeeze(1)], dim=1)
        prediction = self.fc_out(pred_input)  # (batch, vocab_size)
        return prediction, h_n, c_n, attn_weights

# -----------------------
# Seq2Seq wrapper
# -----------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def create_mask(self, src):
        # src: (batch, seq)
        mask = (src != self.pad_idx).to(DEVICE)
        return mask

    def forward(self, src, src_len, trg=None, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # trg: (batch, trg_len)
        batch_size = src.size(0)
        encoder_outputs, (h_n, c_n) = self.encoder(src, src_len)
        # h_n/c_n: (num_layers, batch, dec_hidden) since encoder was bidir and we cat
        max_trg_len = trg.size(1) if trg is not None else MAX_DECODING_LEN
        outputs = torch.zeros(batch_size, max_trg_len, self.decoder.fc_out.out_features).to(DEVICE)
        input_tok = torch.full((batch_size,), vocab.token2idx[SOS_TOKEN], dtype=torch.long).to(DEVICE)
        mask = self.create_mask(src)
        hidden = h_n  # initial decoder hidden
        cell = c_n
        for t in range(0, max_trg_len):
            pred, hidden, cell, _ = self.decoder(input_tok, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = pred
            teacher_force = (random.random() < teacher_forcing_ratio) if trg is not None else False
            top1 = pred.argmax(1)
            if teacher_force and trg is not None:
                input_tok = trg[:, t]  # use ground truth
            else:
                input_tok = top1
        return outputs

# -----------------------
# Training / evaluation helpers
# -----------------------
def train_epoch(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0
    for src, src_len, trg, trg_len in dataloader:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        src_len = src_len.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, src_len, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        # output: (batch, trg_len, vocab)
        # we compute loss for t=1..trg_len-1 (skip initial SOS as target)
        output_dim = output.shape[-1]
        output = output[:, 1:, :].contiguous().view(-1, output_dim)
        target = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, src_len, trg, trg_len in dataloader:
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)
            src_len = src_len.to(DEVICE)
            output = model(src, src_len, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:, :].contiguous().view(-1, output_dim)
            target = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, target)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def greedy_decode(model, src_sentence: str, vocab: Vocab, max_len=MAX_DECODING_LEN):
    model.eval()
    with torch.no_grad():
        src_ids = [vocab.token2idx[SOS_TOKEN]] + vocab.encode(src_sentence) + [vocab.token2idx[EOS_TOKEN]]
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
        encoder_outputs, (h_n, c_n) = model.encoder(src_tensor, src_len)
        hidden, cell = h_n, c_n
        input_tok = torch.tensor([vocab.token2idx[SOS_TOKEN]], dtype=torch.long).to(DEVICE)
        mask = model.create_mask(src_tensor)
        decoded_idxs = []
        for _ in range(max_len):
            pred, hidden, cell, attn_weights = model.decoder(input_tok, hidden, cell, encoder_outputs, mask)
            top1 = pred.argmax(1).item()
            if top1 == vocab.token2idx[EOS_TOKEN]:
                break
            decode
