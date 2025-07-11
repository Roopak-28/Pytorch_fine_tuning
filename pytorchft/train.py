import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

class YelpDataset(Dataset):
    def __init__(self, csv_file, vocab=None, max_len=128):
        self.data = pd.read_csv(csv_file, header=None)
        self.tokenizer = get_tokenizer("basic_english")
        self.max_len = max_len
        if vocab is None:
            self.vocab = build_vocab_from_iterator([self.tokenizer(text) for text in self.data[1]], specials=["<unk>"])
            self.vocab.set_default_index(self.vocab["<unk>"])
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data.iloc[idx, 1], self.data.iloc[idx, 0] - 1
        tokens = self.tokenizer(text)
        indices = [self.vocab[token] for token in tokens[:self.max_len]]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    return torch.stack(texts), torch.stack(labels)

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, n_class)

    def forward(self, x):
        x = self.embedding(x).mean(1)
        return self.fc(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_output', type=str, default="output/yelp_model.pth")
    args = parser.parse_args()

    dataset = YelpDataset(args.data_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = SimpleClassifier(len(dataset.vocab), 100, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print(f"Model saved to {args.model_output}")

if __name__ == "__main__":
    main()
