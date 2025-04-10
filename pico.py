# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        # Call parent constructor (Dataset)
        super().__init__()
        # Store the two input lists of token sequences
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        # Store the probability of sampling from tinystories_seqs
        self.p_tiny = p_tiny

        # Flags to indicate whether each list has any sequences
        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        # Total number of sequences across both datasets
        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        # Raise an error if both lists are empty
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        # Return the total number of sequences
        return self.total_length

    def __getitem__(self, idx):
        # Generate a random float between 0 and 1
        r = random.random()

        # If both datasets are available
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                # Sample from tinystories_seqs with probability p_tiny
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                # Otherwise, sample from other_seqs
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            # If only tinystories are available, sample from it
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            # If only other_seqs are available, sample from it
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        # Convert the selected sequence into a 1D LongTensor
        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    # Step 1: Determine the maximum sequence length in the batch
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    # Step 2: Create a zero-padded tensor of shape (max_len, batch_size)
    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    
    # Step 3: Copy each sequence into the padded tensor column-wise
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    # Return the padded tensor
    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
        Model outputs: raw, unnormalized scores for each vocabulary word.
    tokens: (seq_len, batch)
        Ground-truth tokens corresponding to each timestep and batch element.

    Next-token prediction => we shift target by 1.
    That means: the model’s prediction at position t should match the token at position t+1.
    """
    seq_len, batch_size, vocab_size = logits.shape

    # If sequence is too short to predict next token, return dummy loss
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Remove the last time step from logits
    # These will be used to predict the *next* token
    preds = logits[:-1, :, :]  # shape: (seq_len-1, batch, vocab_size)

    # Remove the first token from the targets
    # These are the actual next tokens
    gold = tokens[1:, :]       # shape: (seq_len-1, batch)

    # Flatten both tensors so we can apply cross-entropy:
    #   preds: (total_positions, vocab_size)
    #   gold: (total_positions,)
    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)

    # Standard classification loss: log_softmax + NLL
    return F.cross_entropy(preds, gold)

# Define the neural network model class, inheriting from PyTorch's base module class nn.Module
class KGramMLPSeqModel(nn.Module):
    """
    This model predicts the next token in a sequence based on the previous 'k' tokens.
    For each position 't' in the input sequence (length seq_len), it does the following:
    1. Gathers the preceding 'k' tokens (context).
    2. Converts these 'k' tokens into dense embeddings using an embedding layer.
    3. Flattens these k embeddings into a single vector.
    4. Passes this vector through a Multi-Layer Perceptron (MLP).
    5. The MLP outputs logits, which represent the likelihood of each token in the vocabulary being the next token at position 't'.

    The final output shape is (seq_len, batch_size, vocab_size), providing logits for each position in the sequence for every item in the batch.

    Note: This approach can consume significant memory, especially with a large vocabulary size (vocab_size)
    or long sequences (seq_len), because it processes each position independently and potentially stores
    intermediate activations. The 'chunk_size' parameter is introduced to process the sequence in smaller
    blocks, reducing peak memory usage by processing only a part of the sequence at a time.
    """

    # The constructor for the model
    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        """
        Initializes the KGramMLPSeqModel.

        Args:
            vocab_size (int): The total number of unique tokens in the vocabulary.
            k (int): The number of preceding tokens (context window size) to use for prediction. Defaults to 3.
            embed_size (int): The dimensionality of the token embeddings. Defaults to 1024.
            num_inner_layers (int): The number of hidden layers in the MLP. Defaults to 1.
            chunk_size (int): The number of sequence steps to process in one go during the forward pass
                              to manage memory usage. Defaults to 1 (process step-by-step).
        """
        # Call the constructor of the parent class (nn.Module)
        super().__init__()

        # Store the configuration parameters as instance variables
        self.k = k  # Context window size
        self.vocab_size = vocab_size  # Size of the vocabulary
        self.embed_size = embed_size  # Dimensionality of token embeddings
        self.num_inner_layers = num_inner_layers  # Number of hidden layers in the MLP
        # Chunk size for processing the sequence to manage memory
        # A larger chunk_size might be faster but uses more memory.
        self.chunk_size = chunk_size

        # --- Define Model Layers ---

        # Embedding layer: Maps discrete token IDs (integers) to dense vectors (embeddings).
        # Input: Token IDs (LongTensor)
        # Output: Corresponding embeddings (FloatTensor) of shape (*, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Note: Using nn.Embedding is more memory-efficient and standard than creating one-hot vectors explicitly.

        # List to hold the layers of the MLP
        layers = []
        # Calculate the input dimension for the MLP: k tokens * embedding size per token
        input_dim = k * embed_size

        # Build the inner (hidden) layers of the MLP
        for _ in range(num_inner_layers):
            # Layer Normalization: Helps stabilize training and improves convergence, especially in deeper networks.
            # It normalizes the activations across the feature dimension (input_dim).
            layers.append(nn.LayerNorm(input_dim))

            # Linear Layer (Fully Connected Layer): Transforms the input features.
            # Input shape: (batch_size, input_dim)
            # Output shape: (batch_size, input_dim)
            linear_layer = nn.Linear(input_dim, input_dim)

            # Weight Initialization (Xavier Uniform): A common technique to initialize weights
            # that helps prevent vanishing or exploding gradients during training, promoting stable learning.
            # It initializes weights based on the layer's input and output dimensions.
            nn.init.xavier_uniform_(linear_layer.weight)

            # Bias Initialization (Zeros): Initializing biases to zero is a standard practice
            # and often works well, preventing the layer from having an initial preference.
            nn.init.zeros_(linear_layer.bias)

            # Add the initialized linear layer to the list
            layers.append(linear_layer)

            # Activation Function (ReLU - Rectified Linear Unit): Introduces non-linearity into the model,
            # allowing it to learn complex patterns. It's computationally efficient (output = max(0, input)).
            layers.append(nn.ReLU())

            # Dropout: A regularization technique to prevent overfitting. During training, it randomly sets
            # a fraction (here 10%) of input units to 0 at each update, forcing the network to learn
            # more robust features. It is automatically disabled during evaluation (model.eval()).
            layers.append(nn.Dropout(0.1))

        # Final Linear Layer (Projection Layer): Maps the final hidden representation to the vocabulary size.
        # Input shape: (batch_size, input_dim)
        # Output shape: (batch_size, vocab_size) - These are the raw logits for each token in the vocab.
        final_layer = nn.Linear(input_dim, vocab_size)

        # Initialize the final layer's weights using Xavier Uniform initialization.
        nn.init.xavier_uniform_(final_layer.weight)
        # Initialize the final layer's biases to zero.
        nn.init.zeros_(final_layer.bias)

        # Add the final projection layer to the list
        layers.append(final_layer)

        # Combine all defined layers into a sequential container (nn.Sequential).
        # This allows data to flow through the layers in the order they were added.
        self.net = nn.Sequential(*layers) # The '*' unpacks the list of layers

    # Defines the computation performed at every call (forward pass)
    def forward(self, tokens_seq):
        """
        Performs the forward pass of the model.

        Args:
            tokens_seq (torch.Tensor): A tensor containing sequences of token IDs.
                                       Expected shape: (seq_len, batch_size)

        Returns:
            torch.Tensor: A tensor containing the output logits for each position in the sequence.
                          Shape: (seq_len, batch_size, vocab_size)
        """
        # Get the sequence length and batch size from the input tensor's shape
        seq_len, batch_size = tokens_seq.shape

        # List to store the outputs (logits) for each chunk of the sequence
        outputs = []

        # --- Process the sequence in chunks to manage memory ---
        # Initialize the starting index for chunk processing
        start = 0
        # Loop through the sequence, processing 'chunk_size' steps at a time
        while start < seq_len:
            # Determine the end index for the current chunk, ensuring it doesn't exceed seq_len
            end = min(start + self.chunk_size, seq_len)

            # List to store outputs for the current chunk
            block_outputs = []

            # --- Iterate through time steps (t) within the current chunk ---
            for t in range(start, end):
                # List to store logits for each item in the batch at time step t
                batch_logits = []

                # --- Iterate through each sequence (b) in the batch ---
                for b in range(batch_size):
                    # --- Gather the context of the last 'k' tokens for position 't' ---
                    # Check if we are near the beginning of the sequence (t < k)
                    if t < self.k:
                        # If not enough preceding tokens exist, pad with zeros from the left.
                        # Calculate how many padding tokens are needed.
                        needed = self.k - t
                        # Create the context by padding with 'needed' zeros and appending the available tokens before t.
                        # Assumes token ID 0 is a suitable padding/start token.
                        # `.tolist()` converts tensor slice to list for concatenation.
                        context_ids = [0] * needed + tokens_seq[:t, b].tolist()
                    else:
                        # If enough preceding tokens exist, take the slice from t-k to t.
                        # `.tolist()` converts tensor slice to list.
                        context_ids = tokens_seq[t - self.k:t, b].tolist()

                    # Convert the list of context token IDs back to a PyTorch tensor.
                    # Ensure it's of type Long (required for nn.Embedding) and on the same device as the input.
                    context_ids = torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device)

                    # --- Get Embeddings and Prepare for MLP ---
                    # Retrieve the embeddings for the context tokens using the embedding layer.
                    # Shape of context_embeds before flatten: (k, embed_size)
                    # Flatten the k embeddings into a single vector.
                    # Shape after flatten: (k * embed_size)
                    # Unsqueeze(0) adds a batch dimension of 1, required by the MLP (nn.Linear expects batch dimension).
                    # Shape after unsqueeze: (1, k * embed_size)
                    context_embeds = self.embedding(context_ids).flatten().unsqueeze(0)

                    # --- Pass through MLP ---
                    # Feed the flattened context embeddings through the sequential MLP network (self.net).
                    # Output shape: (1, vocab_size) - representing logits for the next token prediction.
                    logits_b = self.net(context_embeds)

                    # Append the logits for this batch item to the list for the current time step
                    batch_logits.append(logits_b)

                # --- Combine Batch Results for Time Step t ---
                # Concatenate the logits from all batch items along dimension 0 (the batch dimension).
                # Shape before cat: List of 'batch_size' tensors, each (1, vocab_size)
                # Shape after cat: (batch_size, vocab_size)
                # Unsqueeze(0) adds a sequence dimension of 1 (representing the current time step t).
                # Shape after unsqueeze: (1, batch_size, vocab_size)
                # Append the combined logits for time step t to the block_outputs list.
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))

            # --- Combine Chunk Results ---
            # Concatenate the outputs for all time steps within the current chunk along dimension 0 (the sequence dimension).
            # Shape before cat: List of 'chunk_size' tensors, each (1, batch_size, vocab_size)
            # Shape after cat: (chunk_size, batch_size, vocab_size) - or (actual_chunk_size, ...) if end < start + chunk_size
            block_outputs = torch.cat(block_outputs, dim=0)

            # Append the results of the processed chunk to the main outputs list
            outputs.append(block_outputs)

            # Move the start index to the beginning of the next chunk
            start = end

        # --- Combine All Chunk Outputs ---
        # Concatenate the outputs from all processed chunks along dimension 0 (the sequence dimension).
        # Shape before cat: List of chunk tensors, each (chunk_size, batch_size, vocab_size)
        # Shape after cat: (seq_len, batch_size, vocab_size) - the final desired output shape.
        outputs = torch.cat(outputs, dim=0)

        # Return the final tensor of logits
        return outputs



################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()

        self.vocab_size = vocab_size         # Size of vocabulary (number of unique tokens)
        self.embed_size = embed_size         # Size of token embedding vectors
        self.hidden_size = hidden_size       # Number of hidden units in the LSTM

        # Learnable embedding table: maps token IDs to vectors of size `embed_size`
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM layer:
        #   Input size = embed_size
        #   Hidden state size = hidden_size
        #   batch_first=False => input shape is (seq_len, batch, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)

        # Final linear layer:
        #   Maps LSTM outputs to vocab-size logits for next-token prediction
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
            A sequence of token indices for each batch element.

        Returns:
            logits: (seq_len, batch, vocab_size)
            Unnormalized predictions of the next token at each time step.
        """

        # Step 1: Convert token indices to embeddings
        # Output shape: (seq_len, batch, embed_size)
        emb = self.embedding(tokens_seq)

        # Step 2 (optional but recommended): optimize LSTM weight layout for faster GPU training
        self.lstm.flatten_parameters()

        # Step 3: Run the sequence through the LSTM
        # Output `out`: (seq_len, batch, hidden_size)
        # We discard the second output (_) which contains the final hidden and cell states
        out, _ = self.lstm(emb)

        # Step 4: Map LSTM outputs to logits over the vocabulary
        # Output shape: (seq_len, batch, vocab_size)
        logits = self.linear(out)

        return logits



################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

# Define the RMSNorm (Root Mean Square Normalization) layer
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    This is a normalization technique similar to Layer Normalization but simpler.
    Instead of normalizing based on mean and variance, it normalizes based on the
    root mean square (RMS) of the activations, and then scales the result using
    a learnable parameter (`weight`). It does *not* subtract the mean, making it
    computationally slightly simpler than LayerNorm. It's often used in modern
    Transformer architectures like LLaMA.

    Args:
        dim (int): The dimension of the input features to normalize (typically the embedding dimension).
        eps (float): A small value added to the denominator for numerical stability,
                     preventing division by zero. Defaults to 1e-5.
    """
    def __init__(self, dim, eps=1e-5):
        # Call the constructor of the parent class (nn.Module)
        super().__init__()
        # Store the epsilon value for numerical stability
        self.eps = eps
        # Define a learnable scaling parameter 'weight'.
        # This parameter allows the network to adaptively scale the normalized output.
        # It's initialized to ones, so initially, the normalization effect dominates.
        # nn.Parameter registers this tensor as a model parameter, so it's included
        # in gradient calculations and optimizer updates.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Apply RMS Normalization to the input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor. Normalization is typically applied
                              over the last dimension (the feature dimension).
                              Expected shape: (*, dim)

        Returns:
            torch.Tensor: The normalized and scaled tensor, with the same shape as the input.
        """
        # Calculate the Root Mean Square (RMS) of the input tensor `x`.
        # 1. x.pow(2): Square each element of the input tensor.
        # 2. mean(dim=-1, keepdim=True): Calculate the mean across the last dimension (features).
        #    `keepdim=True` retains the dimension of size 1, making it broadcastable later.
        # 3. add(self.eps): Add the small epsilon value for numerical stability before the square root.
        # 4. sqrt(): Take the square root to get the RMS value.
        # The resulting `norm` tensor has shape (*, 1).
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # Normalize the input tensor `x` by dividing it by its RMS value (`norm`).
        # Due to broadcasting rules and `keepdim=True` above, `norm` divides each feature vector.
        # Then, scale the normalized output by the learnable `self.weight` parameter.
        return self.weight * (x / norm)

# Define the Multi-Head Self-Attention mechanism
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module commonly used in Transformers.

    This module allows the model to weigh the importance of different tokens in the
    input sequence when computing the representation of each token. It does this by
    projecting the input into Query (Q), Key (K), and Value (V) spaces multiple times
    (using different "heads") and performing scaled dot-product attention independently
    for each head. The results from the heads are then concatenated and projected again.

    Args:
        d_model (int): The total dimensionality of the input and output features (embedding dimension).
        n_heads (int): The number of parallel attention heads. `d_model` must be divisible by `n_heads`.
    """
    def __init__(self, d_model, n_heads):
        # Call the constructor of the parent class (nn.Module)
        super().__init__()
        # Ensure that the model dimension is divisible by the number of heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Store the number of attention heads
        self.n_heads = n_heads
        # Calculate the dimension of each attention head's Query, Key, and Value vectors
        self.d_head = d_model // n_heads

        # Define linear layers for transforming the input `x` into Query (Q), Key (K), and Value (V) vectors.
        # These projections allow the model to learn different aspects of the input for attention.
        # `bias=False` is a common choice in transformer implementations, potentially reducing parameters
        # and sometimes improving stability or performance slightly, especially when combined with normalization.
        self.W_q = nn.Linear(d_model, d_model, bias=False) # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False) # Key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False) # Value projection

        # Define the final linear layer to project the concatenated outputs of the attention heads back
        # to the original model dimension `d_model`. `bias=False` is used here too.
        self.W_o = nn.Linear(d_model, d_model, bias=False) # Output projection

        # Define a dropout layer to apply to the attention weights (softmax scores).
        # This helps prevent overfitting by randomly setting some attention weights to zero during training,
        # forcing the model not to rely too heavily on specific token relationships.
        self.dropout = nn.Dropout(0.1) # Dropout probability of 0.1

    def forward(self, x):
        """
        Perform the forward pass for Multi-Head Self-Attention.

        Args:
            x (torch.Tensor): Input tensor.
                              Expected shape: (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor after attention and projection.
                          Shape: (batch_size, seq_len, d_model)
        """
        # Get the input tensor dimensions: Batch size (B), sequence length (T), and model dimension (C = d_model)
        B, T, C = x.shape

        # 1. Project input `x` into Q, K, V using the linear layers.
        #    Resulting shape for q, k, v initially: (B, T, C)
        # 2. Reshape Q, K, V to separate the heads.
        #    - `view(B, T, self.n_heads, self.d_head)` reshapes into (B, T, n_heads, d_head).
        #    - `transpose(1, 2)` swaps the sequence length (T) and head (n_heads) dimensions.
        #    Resulting shape for q, k, v: (B, n_heads, T, d_head). This layout groups computations by head.
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Compute attention scores using scaled dot-product.
        #    - `k.transpose(-2, -1)` transposes the last two dimensions of K (T, d_head) -> (d_head, T).
        #    - `torch.matmul(q, ...)` computes the dot product between Q and transposed K.
        #      Shape: (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        #    - `/ (self.d_head ** 0.5)` scales the scores by the square root of the head dimension.
        #      This scaling prevents the dot products from becoming too large, which could lead to
        #      vanishing gradients in the softmax function.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # 4. Apply causal masking (for autoregressive models like decoders).
        #    This ensures that a position `i` can only attend to positions `j <= i` (itself and previous tokens),
        #    preventing it from "seeing" future tokens during training.
        #    - `torch.ones(T, T, device=x.device)` creates a TxT matrix of ones.
        #    - `torch.tril(...)` creates a lower triangular matrix (keeps elements below and on the diagonal).
        #    - `unsqueeze(0).unsqueeze(0)` adds batch and head dimensions for broadcasting: (1, 1, T, T).
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        #    - `masked_fill(causal_mask == 0, float("-inf"))` replaces scores where the mask is 0 (upper triangle)
        #      with negative infinity. This makes their contribution zero after the softmax.
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # Optional: Extra safety check for NaNs that might arise from computations.
        # Replace any potential NaNs with a very small number before softmax.
        # Often, the causal mask and epsilon in normalization layers prevent this.
        scores = scores.masked_fill(torch.isnan(scores), -1e9) # Using -1e9 as a proxy for -inf

        # 5. Convert raw scores to attention weights using softmax.
        #    Softmax is applied along the last dimension (dim=-1), which corresponds to the keys (source tokens).
        #    This ensures weights for each query position sum to 1. Shape remains (B, n_heads, T, T).
        attn_weights = F.softmax(scores, dim=-1)

        # 6. Apply dropout to the attention weights.
        attn_weights = self.dropout(attn_weights)

        # 7. Compute the weighted sum of the Value vectors using the attention weights.
        #    - `torch.matmul(attn_weights, v)` combines values based on attention.
        #      Shape: (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        attn_output = torch.matmul(attn_weights, v)

        # 8. Merge the attention heads back together.
        #    - `transpose(1, 2)` swaps the head and sequence length dimensions back: (B, T, n_heads, d_head).
        #    - `.contiguous()` ensures the tensor is stored in contiguous memory block, which is required
        #      by `.view()` after certain operations like transpose.
        #    - `view(B, T, C)` reshapes the tensor back to the original input shape (B, T, d_model).
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # 9. Apply the final output linear projection.
        #    Shape: (B, T, d_model) -> (B, T, d_model)
        return self.W_o(attn_output)

# Define a single block of the Transformer
class TransformerBlock(nn.Module):
    """
    A single Transformer block, combining Multi-Head Self-Attention and a Feedforward Network (MLP).

    This block typically includes:
    1. Pre-Normalization: Normalization applied *before* the sub-layer (Attention or MLP).
    2. Multi-Head Self-Attention.
    3. Residual Connection: Adds the input of the sub-layer to its output.
    4. Another Pre-Normalization layer.
    5. A Position-wise Feedforward Network (MLP).
    6. Another Residual Connection.

    Using pre-normalization (Norm -> Sub-layer -> Add) often leads to more stable training
    compared to post-normalization (Sub-layer -> Add -> Norm).

    Args:
        d_model (int): The dimensionality of the input/output features.
        n_heads (int): The number of attention heads for the self-attention module.
    """
    def __init__(self, d_model, n_heads):
        # Call the constructor of the parent class (nn.Module)
        super().__init__()

        # Instantiate the Multi-Head Self-Attention module
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        # Instantiate the first RMSNorm layer (applied before attention)
        self.norm1 = RMSNorm(d_model)
        # Instantiate the second RMSNorm layer (applied before the MLP)
        self.norm2 = RMSNorm(d_model)

        # Define the Feedforward Network (MLP).
        # This usually consists of two linear layers with a non-linearity in between.
        # It processes each position independently (position-wise).
        self.mlp = nn.Sequential(
            # First linear layer: Expands the dimension from d_model to 4 * d_model.
            # This expansion factor of 4 is a common choice in Transformers.
            nn.Linear(d_model, 4 * d_model),
            # Activation function: GELU (Gaussian Error Linear Unit) is often used in modern
            # transformers as it can perform slightly better than ReLU.
            nn.GELU(),
            # Second linear layer: Projects the dimension back from 4 * d_model to d_model.
            nn.Linear(4 * d_model, d_model),
            # Dropout layer for regularization within the MLP.
            nn.Dropout(0.1),
        )

    def forward(self, x):
        """
        Perform the forward pass for a single Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
                              Expected shape: (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor after passing through the block.
                          Shape: (batch_size, seq_len, d_model)
        """
        # --- First Sub-layer: Multi-Head Self-Attention ---
        # 1. Apply RMSNorm (self.norm1) to the input `x`.
        # 2. Pass the normalized tensor through the attention mechanism (self.attn).
        # 3. Add the original input `x` to the output of the attention mechanism (Residual Connection).
        #    This helps with gradient flow and allows the network to easily learn identity mappings.
        x = x + self.attn(self.norm1(x))

        # --- Second Sub-layer: Feedforward Network (MLP) ---
        # 1. Apply RMSNorm (self.norm2) to the output of the first sub-layer.
        # 2. Pass the normalized tensor through the MLP (self.mlp).
        # 3. Add the input to the MLP (output of the first sub-layer) to the output of the MLP (Residual Connection).
        x = x + self.mlp(self.norm2(x))

        # Return the final output of the Transformer block
        return x

# Define the complete Transformer Model
class TransformerModel(nn.Module):
    """
    A complete Transformer model composed of embedding layers, multiple Transformer blocks,
    and a final output layer.

    This specific implementation uses:
    - Token embeddings.
    - Learnable positional embeddings (added to token embeddings).
    - A stack of Transformer blocks.
    - Final normalization (RMSNorm).
    - A linear layer to project the final hidden states to vocabulary logits.

    Args:
        vocab_size (int): The size of the vocabulary. Defaults to 50257 (GPT-2's vocab size).
        d_model (int): The dimensionality of the embeddings and hidden states. Defaults to 1024.
        n_heads (int): The number of attention heads in each Transformer block. Defaults to 2.
        n_blocks (int): The number of Transformer blocks to stack. Defaults to 4.
    """
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4):
        # Call the constructor of the parent class (nn.Module)
        super().__init__()

        # --- Embedding Layers ---
        # Token embedding layer: Maps input token IDs (integers) to dense vectors of size `d_model`.
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional embedding layer: Maps position indices (0 to T-1) to dense vectors of size `d_model`.
        # Since Transformers process tokens in parallel, they don't inherently know the order.
        # Positional embeddings provide this positional information. Using `nn.Embedding` makes them learnable.
        # A fixed size (e.g., 2048) determines the maximum sequence length this model can handle directly.
        self.pos_embedding = nn.Embedding(2048, d_model) # Max sequence length of 2048

        # --- Transformer Blocks ---
        # Create a stack of `n_blocks` Transformer blocks.
        # `nn.ModuleList` is used to store multiple modules correctly, ensuring they are registered
        # as part of the model and their parameters are tracked.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_blocks)
        ])

        # --- Output Layers ---
        # Final normalization layer applied after the last Transformer block.
        self.norm_final = RMSNorm(d_model)
        # Final linear layer (sometimes called the "unembedding" layer or language modeling head)
        # Projects the final `d_model`-dimensional hidden states to `vocab_size` dimensions,
        # producing raw logits for each token in the vocabulary.
        # `bias=False` is often used here, especially if weight tying with the input embedding
        # `self.embedding.weight` is intended (though not explicitly implemented here).
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        """
        Perform the forward pass for the entire Transformer model.

        Args:
            x (torch.Tensor): Input tensor containing token IDs.
                              Expected shape: (batch_size, seq_len) or (B, T)

        Returns:
            torch.Tensor: Output tensor containing logits for each position.
                          Shape: (batch_size, seq_len, vocab_size) or (B, T, vocab_size)
        """
        # Get batch size (B) and sequence length (T) from the input shape
        B, T = x.shape

        # --- Prepare Input Embeddings ---
        # Create position indices: a tensor [0, 1, 2, ..., T-1].
        # `arange(T)` creates the sequence. `device=x.device` ensures it's on the same device (CPU/GPU) as input.
        # `unsqueeze(0)` adds a batch dimension: (1, T).
        # `expand(B, T)` repeats it B times along the first dimension: (B, T).
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        # Get token embeddings from the input token IDs `x`. Shape: (B, T, d_model).
        token_embed = self.embedding(x)
        # Get positional embeddings from the position indices. Shape: (B, T, d_model).
        pos_embed = self.pos_embedding(positions)

        # Add token and positional embeddings element-wise.
        # This combines the token's meaning with its position information.
        x = token_embed + pos_embed # Shape: (B, T, d_model)

        # --- Pass through Transformer Blocks ---
        # Sequentially apply each Transformer block in the `self.blocks` list.
        # The output of one block becomes the input to the next.
        for block in self.blocks:
            x = block(x) # Shape remains (B, T, d_model)

        # --- Generate Output Logits ---
        # Apply the final normalization layer.
        x = self.norm_final(x) # Shape: (B, T, d_model)

        # Apply the final linear layer to project to vocabulary size logits.
        logits = self.unembed(x) # Shape: (B, T, vocab_size)

        # --- Gradient Clipping ---
        # NOTE: `clip_grad_norm_` modifies gradients IN-PLACE and is typically called *after* `loss.backward()`
        # and *before* `optimizer.step()` during the training loop.
        # Placing it here inside the `forward` pass means it's called *before* any gradients exist.
        # This line, as placed here, will likely have NO EFFECT on training, unless gradients
        # from a previous iteration somehow persist (which they shouldn't in standard loops).
        # Its purpose is to prevent exploding gradients by scaling gradients if their overall norm exceeds `max_norm`.
        # It should be moved to the training loop for its intended effect.
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Return the final logits
        return logits



################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    # Apply softmax to the logits to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort the probabilities in descending order and get the indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute the cumulative sum of sorted probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find the smallest k such that the cumulative sum of the top k probs >= p
    k = torch.sum(cumulative_probs < p).item() + 1  # +1 to include the k-th token
    
    # Get the top k indices and their corresponding probabilities
    top_k_indices = sorted_indices[:k]
    top_k_probs = sorted_probs[:k]
    
    # Sample a token from the top k
    sampled_index = torch.multinomial(top_k_probs, 1)
    sampled_token = top_k_indices[sampled_index]
    
    return sampled_token.item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training         # Save the current training/eval mode of the model
    model.eval()                          # Set model to evaluation mode for inference
    with torch.no_grad():                # Disable gradient calculation for efficiency
        context_tokens = enc.encode(init_text)  # Tokenize the initial input text
        annotation_list = []                   # Store monosemantic annotations if needed

        for step_i in range(max_new_tokens):
            # Create input tensor (seq_len, 1) and move to the appropriate device
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)

            # Forward pass: get logits for the entire sequence
            logits_seq = model(seq_tensor)              # (seq_len, 1, vocab_size)

            # Get logits for the last time step only (shape: vocab_size,)
            next_logits = logits_seq[-1, 0, :]

            if top_p is None:
                # Greedy decoding: choose the most likely token
                chosen_token = torch.argmax(next_logits).item()
            else:
                # Top-p (nucleus) sampling: choose from top tokens whose cumulative prob ≤ p
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            # Add chosen token to context for the next round
            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                # Optional: analyze which neurons were responsible for generating this token
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                # If no monosemantic info, just record the token without neighbors
                annotation_list.append((chosen_token, []))

    model.train(was_training)  # Restore original training/eval mode

    # Decode the full generated sequence (initial + generated tokens)
    final_text = enc.decode(context_tokens)

    # Decode only the original input (before generation)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])

    annotated_strs = [prefix_text]  # Start annotated version with prefix

    # Add annotations (e.g., nearest neighbors) to each generated token
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            # Decode each neighbor and add annotation
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    # Join annotated tokens into a string
    annotated_text = "".join(annotated_strs)

    return final_text, annotated_text  # Return both raw and annotated outputs


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a"):
    """
    Train a language model for a number of epochs using a tokenized dataset loader.

    Args:
        model: the language model to train.
        loader: a PyTorch DataLoader yielding batches of token sequences.
        epochs: number of full passes through the dataset.
        model_name: identifier for logging.
        device: where to move data/model (e.g. "cuda" or "cpu").
        lr: learning rate.
        log_steps: how often to log loss stats.
        sample_interval: seconds between generating samples.
        max_steps_per_epoch: optionally cap the number of training steps per epoch.
        enc: tokenizer/decoder (optional, used for generating text samples).
        monosemantic_info: data for monosemantic analysis (optional).
        prompt: initial prompt for generating text during training.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer setup

    start_time = time.time()           # Track when training started
    next_sample_time = start_time      # Time threshold for generating samples
    global_step = 0                    # Total number of steps (across epochs)

    for epoch in range(1, epochs + 1):
        model.train()                 # Enable training mode (dropout, gradients, etc.)
        total_loss = 0.0             # Tracks cumulative loss per epoch
        partial_loss = 0.0           # Tracks loss for logging intervals
        partial_count = 0            # Tracks # of steps for log intervals

        step_in_epoch = 0            # Number of steps completed in current epoch
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # Move batch to GPU/CPU

            logits = model(batch_tokens)            # Forward pass
            loss = compute_next_token_loss(logits, batch_tokens)  # Compute token prediction loss

            optimizer.zero_grad()   # Clear gradients
            loss.backward()         # Backprop
            optimizer.step()        # Update weights

            total_loss += loss.item()         # Add to epoch total
            partial_loss += loss.item()       # Add to rolling total for logging
            partial_count += 1                # Increment for average calculation

            # Logging every `log_steps` steps
            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            # Periodically generate sample text to track model behavior
            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # top-p=1.0 = full sampling from softmax (no cutoff)
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            # Optional early stopping within epoch
            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        # Print average loss for this epoch
        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")


################################################################################
# 9. Main
################################################################################

def main():
    """
    Entry point for training and evaluating multiple language models on a mixed dataset.

    This function performs the following:
    1. Parses command-line arguments to configure model and training settings.
    2. Loads tokenized datasets from HuggingFace's TinyStories and/or custom input files.
    3. Combines datasets with a weighted sampling scheme.
    4. Initializes three models: KGramMLP, LSTM, and Transformer.
    5. Trains each model over a number of epochs using a shared data loader.
    6. Periodically logs loss and generates samples during training using different decoding strategies:
       - Greedy decoding
       - Top-p sampling (p=0.95)
       - Random sampling (top-p=1.0)
    7. After training, performs final text generation using the provided prompt.
    8. Prints final annotated generations for qualitative evaluation.

    Models trained:
        - KGramMLPSeqModel
        - LSTMSeqModel
        - TransformerModel (GPT-like)

    Important arguments expected from the CLI (via argparse):
        - kgram_k: context window size for KGramMLP
        - kgram_chunk_size: chunk size for KGramMLP internal processing
        - embed_size: embedding dimension
        - block_size: max sequence length
        - device_id: CUDA/CPU device string (e.g., "cuda:0" or "cpu")
        - tinystories_weight: probability of sampling from TinyStories vs. other data
        - input_files: optional list of custom text file paths
        - prompt: text prompt to use for generation
        - max_steps_per_epoch: early stopping point for each training epoch
        - num_inner_mlp_layers: depth of the MLP in KGramMLP

    Side Effects:
        - Prints training logs and model outputs to stdout.
        - Uses the `generate_text` function to decode and optionally annotate output.
        - Performs training and evaluation directly, without returning anything.

    Note:
        This function is designed to be used as the main script in a command-line context.
    """
        
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 3
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        n_blocks=6
    ).to(device)

    models = {
        "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
        "kvcache_transformer": transformer,
    }


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt  # <--- Pass the user-specified prompt here
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()