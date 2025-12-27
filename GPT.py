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

class MultiHeadAttention(nn.Module):
    """Multiple Heads of self-attention in parallel"""
    
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
        
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0
    
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0
    
    def forward(self, x, use_cache=False):
        B, T, C = x.shape
        # compue q, k and v vectors
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, C, dim=2)
        # reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        
        if use_cache:
            # kv cache path
            if self.k_cache is None:
                # initialize cache
                self.k_cache = torch.zeros(
                    (B, self.n_heads, context_length, self.head_size),
                    dtype=x.dtype, device=device
                )
                self.v_cache = torch.zeros_like(self.k_cache)
                self.cache_pos = 0
            
            # if cache is full, then reset
            if self.cache_pos + T > context_length:
                self.k_cache.zero_()
                self.v_cache.zero_()
                self.cache_pos = 0
            
            old_k = self.k_cache[:, :, :self.cache_pos]
            old_v = self.v_cache[:, :, :self.cache_pos]
            # use the full cache for attention
            full_k = torch.cat([old_k, k], dim=2)
            full_v = torch.cat([old_v, v], dim=2)
            total_len = self.cache_pos + T
            if T == 1:
                causal_mask = torch.ones(T, total_len, device=device, dtype=torch.bool)
            else:
                causal_mask = torch.tril(torch.ones(T, total_len, device=device, dtype=torch.bool))
            
            weights = (q @ full_k.transpose(-2, -1)) * (self.head_size ** -0.5)
            # apply causal masking
            weights = weights.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            weights = F.softmax(weights, dim=-1)
            weights = self.dropout(weights)
            out = weights @ full_v
            
            # store new K/V in cache
            self.k_cache[:, :, self.cache_pos:self.cache_pos+T] = k
            self.v_cache[:, :, self.cache_pos:self.cache_pos+T] = v
            self.cache_pos += T
        else:
            # standard path
            causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            weights = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
            # apply causal masking
            weights = weights.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            weights = F.softmax(weights, dim=-1)
            weights = self.dropout(weights)
            out = weights @ v # shape: (batch, n_heads, time, head_size)

        out = out.transpose(1, 2).contiguous().view(B, T, C) # (batch, time, channel)
        return self.proj(out)
    
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
        
    def forward(self, x, use_cache=False):
        x = x + self.sa_heads(self.ln1(x), use_cache=use_cache) # Pre-normalization, Residual connection + output from multi-head attention block
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
        self.blocks = nn.ModuleList([Block(num_heads, n_emb) for _ in range(n_layers)])
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
        
    def forward(self, idx, targets=None, use_cache=False, start_pos=0):
        B, T = idx.shape
        
        # Both idx and targets are tensors of integers of size (Batch, Time), idx is the input token
        pos = (start_pos + torch.arange(T, device=device)) % context_length
        token_emb = self.token_embedding_table(idx) # (Batch, Time, Channels), Channels = n_emb
        pos_emb = self.position_embedding_table(pos)
        x = token_emb + pos_emb # Now our input contains both token emb and its positional info
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
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
           
    @torch.no_grad()     
    def generate(self, idx, max_new_tokens):
        # idx is the (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # only feed the last token after first step
            idx_cond = idx[:, -context_length:]
            seq_len = idx.size(1)
            start_pos = max(0, seq_len - context_length)
            # Create initial logits
            logits, _ = self(idx_cond, use_cache=False, start_pos=start_pos)
            # Focus only on the last time step as only that token is predicted
            logits = logits[:, -1, :] # (B, C)
            # Generate probabilities with softmax layer
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample the next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Concatenate it with original idx to generate next input
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        
        return idx
    
    @torch.no_grad()
    def generate_with_cache(self, idx, max_new_tokens):
        self.eval()
        # reset cache
        for block in self.blocks:
            block.sa_heads.reset_cache()
            
        full_seq = idx.clone()
        seq_len = idx.size(1)
        
        # initial decoder fill
        logits, _ = self(idx, use_cache=True, start_pos=0)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat([idx, idx_next], dim=1)
        full_seq = torch.cat([full_seq, idx_next], dim=1)
        seq_len += 1
        
        # generation loop
        for _ in range(1, max_new_tokens):
            # chunking logic
            if seq_len >= context_length:
                # recover the last half of the tokens
                recover_len = context_length // 2
                context_window = full_seq[:, -recover_len:]
                
                # hard reset cache
                for block in self.blocks:
                    block.sa_heads.reset_cache()
                    
                # refill the cache with the window
                # Note - we reset the position ids to 0..recover_len to match the reset cache
                logits, _ = self(context_window, use_cache=True, start_pos=0)
                idx = context_window
                seq_len = recover_len
            
            # we are feeding 1 token. Position is relative to current cache size
            last_token = idx[:, -1:]
            start_pos = seq_len - 1
            
            logits, _ = self(last_token, use_cache=True, start_pos=start_pos)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, idx_next], dim=1)
            full_seq = torch.cat([full_seq, idx_next], dim=1)
            seq_len += 1
        
        return full_seq
    
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
model.train()
# Disable KV cache during training
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
model.eval()
with torch.no_grad():
    output = model.generate_with_cache(context, max_new_tokens=10_000)[0].tolist()
text = decode(output)

with open("more.txt", "w") as f:
    f.write(text)