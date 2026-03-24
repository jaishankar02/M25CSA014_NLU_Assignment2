"""
p2_models.py
============
Character-Level Name Generation using RNN Variants — FROM SCRATCH in PyTorch.

This module implements three sequence models for character-level generation:

1. Vanilla Recurrent Neural Network (RNN)
   - Simple recurrence: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
   - One hidden layer, followed by a linear output layer

2. Bidirectional Long Short-Term Memory (BLSTM)
   - Forward and backward LSTM passes
   - Concatenated hidden states for richer context
   - Gates: input, forget, output, cell
   - Ideal for capturing long-range dependencies in both directions

3. RNN with Basic Attention Mechanism
   - Standard RNN encoder (like Vanilla RNN)
   - Additive (Bahdanau-style) attention over all hidden states
   - Context vector combined with current hidden state for prediction
   - Helps the model focus on relevant parts of the generated sequence

All models are implemented using only nn.Module, nn.Linear, nn.Embedding —
NO use of nn.RNN, nn.LSTM, nn.GRU or any pre-built recurrent modules.

Usage:
    from p2_models import CharVanillaRNN, CharBLSTM, CharRNNAttention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Vanilla RNN — Character-Level
# ============================================================

class VanillaRNNCell(nn.Module):
    """
    A single RNN cell implemented from scratch.

    Computation:
        h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

    This is the fundamental building block — we DO NOT use nn.RNNCell.
    """
    def __init__(self, input_size, hidden_size):
        super(VanillaRNNCell, self).__init__()
        self.hidden_size = hidden_size

        # Input-to-hidden weights
        self.W_ih = nn.Linear(input_size, hidden_size)
        # Hidden-to-hidden weights (recurrence)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize weights using Xavier uniform for stable training
        nn.init.xavier_uniform_(self.W_ih.weight)
        nn.init.xavier_uniform_(self.W_hh.weight)
        nn.init.zeros_(self.W_ih.bias)

    def forward(self, x_t, h_prev):
        """
        Forward pass for one time step.

        Args:
            x_t: (batch_size, input_size) — input at time t
            h_prev: (batch_size, hidden_size) — previous hidden state

        Returns:
            h_t: (batch_size, hidden_size) — new hidden state
        """
        h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))
        return h_t


class CharVanillaRNN(nn.Module):
    """
    Character-Level Vanilla RNN for Name Generation.

    Architecture:
        - Embedding layer: maps character indices to dense vectors
        - Single-layer Vanilla RNN (from-scratch RNNCell)
        - Linear output layer: maps hidden state to character logits

    Hyperparameters:
        vocab_size: Number of unique characters (including special tokens)
        embedding_dim: Size of character embeddings (default: 32)
        hidden_size: Size of the RNN hidden state (default: 128)
        num_layers: Number of stacked RNN layers (default: 1)
        dropout: Dropout probability between layers (default: 0.1)
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=128,
                 num_layers=1, dropout=0.1):
        super(CharVanillaRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Stack of from-scratch RNN cells
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embedding_dim if i == 0 else hidden_size
            self.rnn_cells.append(VanillaRNNCell(input_dim, hidden_size))

        # Dropout between layers
        self.dropout = nn.Dropout(dropout)

        # Output projection: hidden state -> character logits
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the entire sequence.

        Args:
            x: (batch_size, seq_len) — character indices
            hidden: list of (batch_size, hidden_size) tensors for each layer,
                    or None to initialize with zeros

        Returns:
            output: (batch_size, seq_len, vocab_size) — logits for each position
            hidden: list of (batch_size, hidden_size) — final hidden states
        """
        batch_size, seq_len = x.size()

        # Initialize hidden states if not provided
        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                      for _ in range(self.num_layers)]

        # Embed input characters
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        outputs = []
        for t in range(seq_len):
            inp = emb[:, t, :]  # (batch_size, embedding_dim)

            new_hidden = []
            for layer_idx, rnn_cell in enumerate(self.rnn_cells):
                h = rnn_cell(inp, hidden[layer_idx])
                new_hidden.append(h)
                inp = self.dropout(h) if layer_idx < self.num_layers - 1 else h

            hidden = new_hidden
            outputs.append(inp)  # Last layer's hidden state

        # Stack outputs: (batch_size, seq_len, hidden_size)
        output = torch.stack(outputs, dim=1)

        # Project to vocabulary: (batch_size, seq_len, vocab_size)
        logits = self.fc_out(output)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden states to zeros."""
        return [torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)]


# ============================================================
# 2. Bidirectional LSTM (BLSTM) — From Scratch
# ============================================================

class LSTMCell(nn.Module):
    """
    A single LSTM cell implemented from scratch.

    LSTM gates (all implemented via linear transformations):
        - Forget gate:  f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
        - Input gate:   i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
        - Cell update:  g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)
        - Output gate:  o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
        - Cell state:   c_t = f_t * c_{t-1} + i_t * g_t
        - Hidden state: h_t = o_t * tanh(c_t)

    We compute all four gates in a single matrix multiply for efficiency,
    then split the result.
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # Combined weight matrix for all 4 gates (efficiency optimization)
        # Input: [x_t, h_{t-1}] → 4 * hidden_size outputs
        self.W_combined = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        # Initialization: use orthogonal for recurrent weights, Xavier for input
        nn.init.xavier_uniform_(self.W_combined.weight[:, :input_size])
        nn.init.orthogonal_(self.W_combined.weight[:, input_size:])

        # Bias initialization: set forget gate bias to 1.0 for better gradient flow
        nn.init.zeros_(self.W_combined.bias)
        with torch.no_grad():
            self.W_combined.bias[hidden_size:2*hidden_size].fill_(1.0)

    def forward(self, x_t, h_prev, c_prev):
        """
        Forward pass for one time step.

        Args:
            x_t: (batch_size, input_size) — input at time t
            h_prev: (batch_size, hidden_size) — previous hidden state
            c_prev: (batch_size, hidden_size) — previous cell state

        Returns:
            h_t: (batch_size, hidden_size) — new hidden state
            c_t: (batch_size, hidden_size) — new cell state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)

        # Compute all gates in one go
        gates = self.W_combined(combined)

        # Split into individual gates
        i_gate = torch.sigmoid(gates[:, :self.hidden_size])                    # Input gate
        f_gate = torch.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])  # Forget gate
        g_gate = torch.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])   # Cell candidate
        o_gate = torch.sigmoid(gates[:, 3*self.hidden_size:])                  # Output gate

        # Update cell state and hidden state
        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * torch.tanh(c_t)

        return h_t, c_t


class CharBLSTM(nn.Module):
    """
    Character-Level Bidirectional LSTM for Name Generation.

    Architecture:
        - Embedding layer: character indices → dense vectors
        - Forward LSTM: processes sequence left-to-right
        - Backward LSTM: processes sequence right-to-left
        - Concatenation: forward + backward hidden states
        - Linear output: 2*hidden_size → vocab_size

    Note: For generation (autoregressive), we primarily use the forward direction.
    The backward pass helps during training with teacher forcing.

    Hyperparameters:
        vocab_size: Number of unique characters
        embedding_dim: Character embedding dimension (default: 32)
        hidden_size: Hidden state size per direction (default: 128)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=128, dropout=0.1):
        super(CharBLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Forward and backward LSTM cells (from scratch)
        self.forward_cell = LSTMCell(embedding_dim, hidden_size)
        self.backward_cell = LSTMCell(embedding_dim, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer: concatenated bidirectional hidden → vocab logits
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass (bidirectional).

        Args:
            x: (batch_size, seq_len) — character indices
            hidden: ignored (initialized internally for each direction)

        Returns:
            output: (batch_size, seq_len, vocab_size) — logits
            hidden: tuple of final (h_fwd, c_fwd, h_bwd, c_bwd)
        """
        batch_size, seq_len = x.size()
        device = x.device

        # Embed
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Initialize states
        h_fwd = torch.zeros(batch_size, self.hidden_size, device=device)
        c_fwd = torch.zeros(batch_size, self.hidden_size, device=device)
        h_bwd = torch.zeros(batch_size, self.hidden_size, device=device)
        c_bwd = torch.zeros(batch_size, self.hidden_size, device=device)

        # Forward pass: left to right
        fwd_outputs = []
        for t in range(seq_len):
            h_fwd, c_fwd = self.forward_cell(emb[:, t, :], h_fwd, c_fwd)
            fwd_outputs.append(h_fwd)

        # Backward pass: right to left
        bwd_outputs = []
        for t in range(seq_len - 1, -1, -1):
            h_bwd, c_bwd = self.backward_cell(emb[:, t, :], h_bwd, c_bwd)
            bwd_outputs.insert(0, h_bwd)

        # Stack and concatenate
        fwd_out = torch.stack(fwd_outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        bwd_out = torch.stack(bwd_outputs, dim=1)
        combined = torch.cat([fwd_out, bwd_out], dim=2)  # (batch_size, seq_len, 2*hidden_size)
        combined = self.dropout(combined)

        # Project to vocab
        logits = self.fc_out(combined)
        return logits, (h_fwd, c_fwd, h_bwd, c_bwd)

    def generate_forward_only(self, x, hidden=None):
        """
        Forward-only pass for autoregressive generation.
        During generation, we can only use the forward direction.

        Args:
            x: (batch_size, seq_len) — input characters
            hidden: tuple (h, c) or None

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden: (h, c) — final forward hidden state
        """
        batch_size, seq_len = x.size()
        device = x.device

        emb = self.embedding(x)

        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h, c = hidden

        outputs = []
        for t in range(seq_len):
            h, c = self.forward_cell(emb[:, t, :], h, c)
            outputs.append(h)

        out = torch.stack(outputs, dim=1)

        # Use only forward hidden size — we need to project through a separate layer
        # For generation, use a simpler linear projection
        # We reuse half of fc_out weights conceptually, but for simplicity use a dedicated layer
        logits = self.fc_out(torch.cat([out, torch.zeros_like(out)], dim=2))
        return logits, (h, c)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))


# ============================================================
# 3. RNN with Basic Attention Mechanism — From Scratch
# ============================================================

class CharRNNAttention(nn.Module):
    """
    Character-Level RNN with Basic (Additive/Bahdanau) Attention.

    Architecture:
        - Embedding layer
        - Vanilla RNN encoder (from scratch, like model 1)
        - Additive attention mechanism:
            score(h_t, h_s) = v^T tanh(W_1 h_t + W_2 h_s)
            alpha = softmax(scores)
            context = sum(alpha * h_s)
        - Concatenation: [context, h_t] → prediction
        - Linear output layer

    The attention allows the model to "look back" at previously generated
    characters and weigh their importance for predicting the next one.

    Hyperparameters:
        vocab_size: Number of unique characters
        embedding_dim: Character embedding dimension (default: 32)
        hidden_size: RNN hidden state size (default: 128)
        attention_size: Attention projection size (default: 64)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=128,
                 attention_size=64, dropout=0.1):
        super(CharRNNAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN cell (from scratch)
        self.rnn_cell = VanillaRNNCell(embedding_dim + hidden_size, hidden_size)
        # Note: input to RNN is [embedding, context_vector]

        # Attention mechanism (Bahdanau-style additive attention)
        self.W_query = nn.Linear(hidden_size, attention_size, bias=False)  # Current hidden
        self.W_key = nn.Linear(hidden_size, attention_size, bias=False)    # Encoder hiddens
        self.v_attn = nn.Linear(attention_size, 1, bias=False)             # Score vector

        # Output projection: [context + hidden] → vocab
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize attention weights
        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_key.weight)
        nn.init.xavier_uniform_(self.v_attn.weight)

    def compute_attention(self, h_current, all_hiddens, mask=None):
        """
        Compute additive (Bahdanau) attention weights.

        Args:
            h_current: (batch_size, hidden_size) — current decoder hidden state
            all_hiddens: (batch_size, num_steps, hidden_size) — all previous hidden states
            mask: (batch_size, num_steps) — optional mask for valid positions

        Returns:
            context: (batch_size, hidden_size) — weighted sum of hidden states
            attn_weights: (batch_size, num_steps) — attention weights
        """
        # Project query: (batch_size, 1, attention_size)
        query = self.W_query(h_current).unsqueeze(1)

        # Project keys: (batch_size, num_steps, attention_size)
        keys = self.W_key(all_hiddens)

        # Compute scores: (batch_size, num_steps, 1) → (batch_size, num_steps)
        scores = self.v_attn(torch.tanh(query + keys)).squeeze(2)

        # Apply mask if provided (mask out padding positions)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=1)

        # Weighted sum: (batch_size, hidden_size)
        context = torch.bmm(attn_weights.unsqueeze(1), all_hiddens).squeeze(1)

        return context, attn_weights

    def forward(self, x, hidden=None):
        """
        Forward pass with attention.

        Args:
            x: (batch_size, seq_len) — character indices
            hidden: (batch_size, hidden_size) or None

        Returns:
            output: (batch_size, seq_len, vocab_size) — logits
            hidden: (batch_size, hidden_size) — final hidden state
        """
        batch_size, seq_len = x.size()
        device = x.device

        # Embed
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Initialize hidden state
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h = hidden

        # Store all hidden states for attention
        all_hiddens = []
        outputs = []

        # Initial context vector (zeros for first step)
        context = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_len):
            # Input is concatenation of embedding and context vector
            rnn_input = torch.cat([emb[:, t, :], context], dim=1)

            # RNN step
            h = self.rnn_cell(rnn_input, h)
            all_hiddens.append(h)

            # Compute attention over all previous hidden states
            if len(all_hiddens) > 1:
                # Stack all hidden states so far: (batch_size, t+1, hidden_size)
                stacked = torch.stack(all_hiddens, dim=1)
                context, _ = self.compute_attention(h, stacked)
            else:
                # First step: context is just the current hidden state
                context = h

            # Combine hidden state and context for output
            combined = torch.cat([h, context], dim=1)
            combined = self.dropout(combined)

            # Project to vocabulary logits
            logit = self.fc_out(combined)
            outputs.append(logit)

        # Stack outputs: (batch_size, seq_len, vocab_size)
        output = torch.stack(outputs, dim=1)

        return output, h

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device=device)


# ============================================================
# Utility: Print model architecture summary
# ============================================================

def print_model_summary(model, model_name):
    """
    Print a clear summary of the model architecture, including
    number of trainable parameters and hyperparameters.
    """
    print(f"\n{'=' * 60}")
    print(f"MODEL ARCHITECTURE: {model_name}")
    print(f"{'=' * 60}")

    total_params = 0
    print(f"\n  Layer Details:")
    print(f"  {'-' * 50}")
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        print(f"    {name:40s} — {str(list(param.shape)):20s} [{num:,} params]")

    print(f"\n  {'=' * 50}")
    print(f"  Total Trainable Parameters: {total_params:,}")
    print(f"  {'=' * 50}")

    # Print hyperparameters
    print(f"\n  Hyperparameters:")
    if hasattr(model, 'vocab_size'):
        print(f"    Vocabulary size:   {model.vocab_size}")
    if hasattr(model, 'embedding_dim'):
        print(f"    Embedding dim:     {model.embedding_dim}")
    if hasattr(model, 'hidden_size'):
        print(f"    Hidden size:       {model.hidden_size}")
    if hasattr(model, 'num_layers'):
        print(f"    Number of layers:  {model.num_layers}")
    if hasattr(model, 'attention_size'):
        print(f"    Attention size:    {model.attention_size}")
    print()

    return total_params
