import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available else 'cpu'
context_length = 32
batch_size = 8
n_embd = 64
dropout = 0.2
n_heads = 4
n_layers = 4
max_iters = 5_000
eval_iters = 200
eval_interval = 500
lr = 3e-4

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        weights = q @ k.transpose(-2, -1) * k.shape[-1] **-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.sa_heads = nn.ModuleList(Head(head_size) for _ in range(n_heads))
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.sa_heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out 
    
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_heads, n_embd):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForwardNetwork()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(*[Block(n_heads, n_embd) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = 0
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -context_length:]
            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
    
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i,ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
train_len = int(0.9 * len(data))
train_data = data[:train_len]
val_data = data[train_len:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in idxs])
    y = torch.stack([data[i+1:i+1+context_length] for i in idxs])
    x = x.to(device)
    y = y.to(device)
    return x, y

model = GPTLanguageModel()
model.to(device)
print(f">> {sum(p.numel() for p in model.parameters())/1e6}M parameters")

def evaluate_loss():
    output = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[iter] = loss
        output[split] = losses.mean()
    model.train()
    return output
    
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
for iter in range(max_iters):
    if iter%eval_interval == 0 or iter == max_iters-1:
        losses = evaluate_loss()
        train_loss, val_loss = losses['train'], losses['val']
        print(f">> Step: {iter} - train_loss: {train_loss}, val_loss: {val_loss}")
    
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\n----- Training Complete -----\n")

prompt = "First Citizen:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=10_000)[0].tolist()
text = decode(output)

with open("more_practice.txt", "w") as f:
    f.write(text)