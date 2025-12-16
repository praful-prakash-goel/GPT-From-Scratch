# Decoder-Only GPT Model from Scratch (NanoGPT-style)

This repository contains a complete, from-scratch implementation of a **decoder-only GPT** language model in PyTorch, heavily inspired by Andrej Karpathy's "nanoGPT" and "Let's build GPT" lectures.

The model is trained character-by-character on any text file (`input.txt`) and can generate coherent text after training.

## Model Architecture

This is a **decoder-only transformer** similar to the original GPT models:

- **Vocabulary Size**: Determined automatically from the input text (character-level tokenization)
- **Embedding Dimension**: `n_emb = 384`
- **Context Length**: `context_length = 256` tokens
- **Number of Layers**: `n_layers = 6`
- **Number of Attention Heads**: `num_heads = 6` (head size = 384 / 6 = 64)
- **Dropout**: `0.2`
- **Feed-Forward Hidden Size**: 4 × embedding dim = 1536 (standard GPT scaling)

### Key Components

1. **Token + Position Embeddings**
   - Learned token embedding table: `(vocab_size, n_emb)`
   - Learned positional embedding table: `(context_length, n_emb)`

2. **Transformer Blocks** (6 stacked)
   Each block consists of:
   - Pre-norm **LayerNorm**
   - **Multi-Head Self-Attention** (causal/masked)
   - Residual connection
   - Pre-norm **LayerNorm**
   - **Feed-Forward Network** (Linear → ReLU → Linear → Dropout)
   - Residual connection

3. **Final LayerNorm** + **Language Modeling Head**
   - Linear projection from `n_emb` → `vocab_size` to produce logits

### Total Parameters
Approximately **~10.8 million** parameters (exact count printed at runtime).

## Features

- Causal (masked) self-attention using lower triangular mask
- Pre-layer normalization (modern GPT style)
- Residual connections
- Scaled dot-product attention
- Proper weight initialization (normal with std=0.02)
- AdamW optimizer with learning rate `3e-4`
- Training with cross-entropy loss
- Train/validation split (90/10)
- Evaluation of loss on both splits every 500 steps
- **Best checkpoint saving** based on validation loss
- Text generation using sampling (multinomial over softmax)
- Automatic resume from best checkpoint if exists

## Requirements

- Python 3.8+
- PyTorch (tested with 2.0+)
- CUDA-capable GPU recommended (falls back to CPU)

No additional packages required.

## Usage

1. Prepare your training text:
   Place your training text in a file named `input.txt` in the same directory.

   Example: Shakespeare's works, books, code, etc.

2. Run the training script (save the Python code as `gpt.py`):
    ```bash
   python gpt.py
  The script will:

- Load `input.txt`
- Build character-level vocabulary
- Train the model for 5000 iterations
- Save the best checkpoint as `best_checkpoint.pt`
- After training, load the best model
- Generate ~10,000 characters starting from prompt `"First Citizen:"`
- Save generated text to `more.txt`

Generated output will be saved to `more.txt`.

### Customization

You can easily modify hyperparameters at the top of the file:

```python
batch_size = 64
context_length = 256
n_emb = 384
n_layers = 6
num_heads = 6
dropout = 0.2
learning_rate = 3e-4
max_iters = 5_000
eval_interval = 500
```
For larger models, increase n_emb, n_layers, num_heads, and context_length accordingly.
You can also change the generation prompt or max_new_tokens near the end of the script.

### Example Output

After training on Shakespeare, the model typically produces coherent Shakespearean-style text:
```text
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.
...
```
(Actual quality improves with longer training and larger models.)

### Notes
- This is a character-level model (not BPE like GPT-2/3).
- Training is relatively fast on modern GPUs (a few hours for decent quality on Shakespeare dataset).
- For better performance, use longer training, larger model, or bigger/more diverse dataset.
- The model uses pre-norm (LayerNorm before attention/FFN) which is the modern standard.
- Checkpointing ensures you always keep the best-performing model on validation loss.

### Credits
Inspired by:

- Andrej Karpathy's "[Let's build GPT](https://youtu.be/kCc8FmEb1nY)" lecture
- nanoGPT repository: https://github.com/karpathy/nanoGPT
