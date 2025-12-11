import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed("1337")
checkpoint_path = "best_checkpoint.pt"

# Hyperparameters
batch_size = 64
context_length = 256
n_emb = 384
n_layers = 6
num_heads = 6
dropout=0.2
learning_rate = 3e-4
max_iters = 5_000
eval_iters = 200
eval_interval = 500

class Head(nn.Module):
    """One head of self attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # input of size (Batch, Time, Channels)
        # output of size (Batch, Time, Head Size)
        B, T, C = x.shape
        # Get the key, query and value vectors
        k = self.key(x) # (B, T, HS)
        q = self.query(x) # (B, T, HS)
        v = self.value(x) # (B, T, HS)
        
        # Compute attention scores using scaled dot-product
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, HS) @ (B, HS, T) --> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, -1)
        weights = self.dropout(weights)
        
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple Heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(head_size * num_heads, n_emb)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, head_size * num_heads)
        out = self.proj(out) # (B, T, n_emb)
        out = self.dropout(out)
        return out
    
class FeedForwardNetwork(nn.Module):
    """A simple linear layer followed by non-linear layer"""
    
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb), # Projection layer
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer Block: Communication followed by computation"""
    
    def __init__(self, num_heads, n_emb):
        super().__init__()
        head_size = n_emb // num_heads
        # Creating a multi-head attention layer
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        # Creating a feed-forward neural network
        self.ffwd = FeedForwardNetwork(n_emb)
        # Creating layer norm
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        
    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # Pre-normalization, Residual connection + output from multi-head attention block
        x = x + self.ffwd(self.ln2(x)) # Pre-normalization, Residual connection + output from ffwd network
        return x
        
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads the embedding for the next token from a look up table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        # Creating a position embedding table which will store positional information
        self.position_embedding_table = nn.Embedding(context_length, n_emb)
        # Creating transformer blocks
        self.blocks = nn.Sequential(*[Block(num_heads, n_emb) for _ in range(n_layers)])
        # Final Layer norm layer
        self.ln_f = nn.LayerNorm(n_emb)
        # Creating a linear layer for getting logits from embeddings
        self.lm_head = nn.Linear(n_emb, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Both idx and targets are tensors of integers of size (Batch, Time), idx is the input token
        token_emb = self.token_embedding_table(idx) # (Batch, Time, Channels), Channels = n_emb
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb # Now our input contains both token emb and its positional info
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (Batch, Time, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # Shape = (N, C) -> N = No. of samples (each with C scores)
            targets = targets.view(B*T) # Shape = (N,) -> N class indices (0, C-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is the (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx upto context length
            idx_cond = idx[:, -context_length:]
            # Create initial logits
            logits, _ = self(idx_cond)
            # Focus only on the last time step as only that token is predicted
            logits = logits[:, -1, :] # (B, C)
            # Generate probabilities with softmax layer
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample the next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Concatenate it with original idx to generate next input
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
        
    
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
# Generating vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating mapping to tokenize characters
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Tokenizing the complete text data
data = torch.tensor(encode(text), dtype=torch.long)
# Splitting into train and validation data
train_len = int(0.9 * len(data))
train_data = data[:train_len]
val_data = data[train_len:]

# Generate batches of inputs
def get_batch(split):
    data = train_data if split == "train" else val_data
    idxs = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in idxs])
    y = torch.stack([data[i+1:i+context_length+1] for i in idxs])
    x, y = x.to(device), y.to(device)
    
    return x, y

model = GPTLanguageModel(vocab_size=vocab_size)
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, "M Parameters")

@torch.no_grad()
def estimate_loss():
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

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Saving best checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['val_loss']
    print(f"Restored model from checkpoint at step: {checkpoint['iter']}")
else:
    best_val_loss = float('inf')
    
# Training loop
for iter in range(max_iters):
    # Evaluate model after a particular interval
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        train_loss, val_loss = losses['train'], losses['val']
        print(f"Step {iter}: train_loss: {train_loss}, val_loss: {val_loss}")
        
        # Saving best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "iter": iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at step {iter}, train_loss : {train_loss}, val_loss : {val_loss}")
        
    # Sample a batch from training data
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n----- Training Complete -----\n")

# Loading best checkpoint
if os.path.exists(checkpoint_path):
    model = GPTLanguageModel(vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded the best model from step {checkpoint['iter']}, val_loss = {checkpoint['val_loss']}, train_loss = {checkpoint['train_loss']}")
    model.eval()

# Generating from model
prompt = "First Citizen:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=10_000)[0].tolist()
text = decode(output)

with open("more.txt", "w") as f:
    f.write(text)