# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import re

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

    Args:
        dim (int): The dimension of the input features to normalize (typically the embedding dimension).
        eps (float): A small value added to the denominator for numerical stability,
                     preventing division by zero. Defaults to 1e-5.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)

# Define the Multi-Head Self-Attention mechanism
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module commonly used in Transformers.

    Args:
        d_model (int): The total dimensionality of the input and output features (embedding dimension).
        n_heads (int): The number of parallel attention heads. `d_model` must be divisible by `n_heads`.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False) # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False) # Key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False) # Value projection
        self.W_o = nn.Linear(d_model, d_model, bias=False) # Output projection

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
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        scores = scores.masked_fill(torch.isnan(scores), -1e9) # Using -1e9 as a proxy for -inf

        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.W_o(attn_output)

# Define a single block of the Transformer

class Swiglu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # Split last dim in half
        return x1 * F.silu(x2)       # Swiglu activation

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * 4 * d_model),  # double hidden dim for chunking
            Swiglu(),
            nn.Linear(4 * d_model, d_model),      # project back to d_model
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Define the complete Transformer Model
class TransformerModel(nn.Module):
    """
    A complete Transformer model composed of embedding layers, multiple Transformer blocks,
    and a final output layer.

    Args:
        vocab_size (int): The size of the vocabulary. Defaults to 50257 (GPT-2's vocab size).
        d_model (int): The dimensionality of the embeddings and hidden states. Defaults to 1024.
        n_heads (int): The number of attention heads in each Transformer block. Defaults to 2.
        n_blocks (int): The number of Transformer blocks to stack. Defaults to 4.
    """
    def __init__(self, vocab_size=50257, d_model=1280, n_heads=10, n_blocks=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_blocks)
        ])
        self.norm_final = nn.LayerNorm(d_model)
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
        B, T = x.shape

        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        token_embed = self.embedding(x)
        pos_embed = self.pos_embedding(positions)

        x = token_embed + pos_embed # Shape: (B, T, d_model)

        for block in self.blocks:
            x = block(x) # Shape remains (B, T, d_model)

        x = self.norm_final(x) # Shape: (B, T, d_model)

        logits = self.unembed(x) # Shape: (B, T, vocab_size)

        return logits



################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################
import torch
import torch.nn.functional as F
import re # Keep for potential other uses, but not for the removed penalty
import time
import torch.optim as optim # Assuming optim is needed for training part

# Helper function (if you are using top-p) - ensure you have one
import torch
import torch.nn.functional as F
# Make sure you have your nucleus_sampling function defined as well if using top_p < 1.0
# from your_module import nucleus_sampling # Or define it here if not imported

# Placeholder for the sampling function if not defined elsewhere
def nucleus_sampling(logits, p):
    """Applies nucleus sampling (top-p) to logits."""
    if p <= 0.0 or p > 1.0:
        # Fallback to greedy if p is invalid or effectively disables sampling diversity
        return torch.argmax(logits).item()
    if p == 1.0:
         # If p is 1, it's equivalent to full sampling from softmax
         probs = F.softmax(logits, dim=-1)
         # Handle case where all logits might be -inf -> NaN probabilities
         if torch.isnan(probs).all():
              print("Warning: All probabilities are NaN in full sampling. Falling back to greedy.")
              return torch.argmax(logits).item()
         return torch.multinomial(probs, 1).item()
    else:
        # Sort logits in descending order and get corresponding probabilities
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold p
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Get the indices of tokens to remove in the original ordering
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # Create a mask or directly modify logits
        logits_clone = logits.clone() # Avoid modifying original logits if needed elsewhere
        logits_clone[indices_to_remove] = -float('inf') # Set logits of removed tokens to -inf

        # Handle case where all logits might be -inf after filtering
        if torch.isinf(logits_clone).all():
             print("Warning: All logits became -inf after top-p filtering. Falling back to greedy on original logits.")
             return torch.argmax(logits).item()

        probs = F.softmax(logits_clone, dim=-1)
        chosen_token = torch.multinomial(probs, 1).item()
        return chosen_token

# --- Full generate_text function with Q, !, and A: penalties ---
def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  temperature=1.0,
                  monosemantic_info=None, # Keep for compatibility if needed elsewhere
                  do_monosemantic=False,  # Keep for compatibility
                  penalty_value=float('inf'), # How much to penalize (inf = forbid)
                  apply_penalties=True): # Flag to enable/disable penalties
    """
    Generates text token by token, applying logit penalties to discourage:
      - Repeating '!' immediately (avoids '!!')
      - Generating ':' immediately after 'A' (avoids 'A:')
      - Repeating 'Q' immediately (avoids 'QQ')

    - Penalties are applied *before* sampling the next token.
    - Assumes '!', 'A', ':', 'Q' are single tokens in the tokenizer 'enc'.
    """
    was_training = model.training
    model.eval()

    # Get token IDs for penalty logic (do this once outside the loop)
    # IMPORTANT: This assumes '!', 'A', ':', 'Q' are encoded as single tokens.
    # If your tokenizer splits them (e.g., BPE), this logic needs adjustment.
    exclaim_token_id, a_token_id, colon_token_id, q_token_id = None, None, None, None
    has_penalty_tokens = False
    local_apply_penalties = apply_penalties # Use a local copy of the flag

    if local_apply_penalties: # Only try to get IDs if penalties are requested
        try:
            # Use enc.encode('X', allowed_special='all') if needed by your tokenizer
            exclaim_token_id = enc.encode('!')[0]
            a_token_id = enc.encode('A')[0]
            colon_token_id = enc.encode(':')[0]
            q_token_id = enc.encode('Q')[0]  # *** Get Q token ID ***
            has_penalty_tokens = True
            # Optional Debugging print:
            # print(f"Debug: Penalty token IDs found: !={exclaim_token_id}, A={a_token_id}, :={colon_token_id}, Q={q_token_id}")
        except IndexError:
            print("Warning: Could not encode one or more penalty tokens ('!', 'A', ':', 'Q') as single tokens. Penalties disabled for this call.")
            local_apply_penalties = False # Disable penalties for this run if tokens aren't found/single
        except Exception as e:
            print(f"Warning: Error getting penalty token IDs: {e}. Penalties disabled for this call.")
            local_apply_penalties = False

    with torch.no_grad():
        try:
            context_tokens = enc.encode(init_text)
            # Ensure context_tokens is a list of integers
            if not isinstance(context_tokens, list):
                # Attempt conversion based on common types, adjust if needed
                if hasattr(context_tokens, 'tolist'):
                     context_tokens = context_tokens.tolist()
                else:
                     # Fallback or raise error if conversion unclear
                     raise TypeError(f"Unsupported type for context_tokens: {type(context_tokens)}")
        except Exception as e:
            print(f"Error encoding initial text: {e}")
            return "Error: Could not encode prompt.", "Error: Could not encode prompt."


        generated_token_data = [] # Store (token_id, neighbors) for generated tokens

        for step_i in range(max_new_tokens):
            if not context_tokens: # Safety check in case encoding failed or list became empty
                print("Warning: context_tokens became empty during generation.")
                break

            # Prepare input tensor
            try:
                seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1) # (seq_len, 1)
            except Exception as e:
                 print(f"Error creating tensor from context_tokens at step {step_i}: {e}")
                 print(f"Context tokens causing error: {context_tokens}")
                 break # Stop generation if tensor creation fails


            # Get logits for the next token
            try:
                logits_seq = model(seq_tensor)      # (seq_len, 1, vocab_size)
                next_logits = logits_seq[-1, 0, :]  # Shape: (vocab_size,)
            except Exception as e:
                print(f"Error during model forward pass at step {step_i}: {e}")
                break # Stop generation if model fails

            # --- ADD: Logit Boosting for A: after ? ---
            boost_value = 5.0 # Adjust as needed (higher = stronger boost)
            # Check local_apply_penalties which might have been disabled if token IDs weren't found
            if local_apply_penalties and has_penalty_tokens and context_tokens:
                try:
                    # Get token ID for the end-of-question marker (e.g., '?')
                    # Adjust if your questions end differently (e.g., newline token)
                    qmark_token_id = enc.encode('?')[0] # Assumes '?' is the marker
                    if context_tokens[-1] == qmark_token_id:
                        # Boost 'A' and ':' logits if the last token was '?'
                        if a_token_id is not None:
                            # print(f"Debug: Boosting 'A' ({a_token_id}) logit.") # Optional Debug
                            next_logits[a_token_id] += boost_value
                        if colon_token_id is not None:
                            # print(f"Debug: Boosting ':' ({colon_token_id}) logit.") # Optional Debug
                            next_logits[colon_token_id] += boost_value # Boost : as well, assuming A: is desired
                except IndexError:
                    # Handle if '?' is not a single token or not found
                    # print("Warning: Could not get single token ID for '?'. Boosting disabled.") # Optional Debug
                    pass # Silently ignore if '?' token isn't setup correctly
                except Exception as e:
                    # print(f"Warning: Error during boosting logic: {e}") # Optional Debug
                    pass # Silently ignore other errors during boosting
            # --- END: Logit Boosting ---
            # --- Apply Temperature ---
            if temperature != 1.0 and temperature > 0:
                next_logits = next_logits / temperature
            elif temperature <= 0:
                 # Using temp=1.0 is safer than potentially causing issues with 0 or negative
                 pass # Effectively use temperature 1.0


            # --- Apply Logit Penalties ---
            # Check local_apply_penalties which might have been disabled if token IDs weren't found
            if local_apply_penalties and has_penalty_tokens and context_tokens:
                last_token = context_tokens[-1]

                # 1. Penalize '!' if the last token was '!'
                if last_token == exclaim_token_id:
                    # print(f"Debug: Penalizing '!' token (ID: {exclaim_token_id})") # Optional Debug
                    next_logits[exclaim_token_id] -= penalty_value

                # 2. Penalize ':' if the last token was 'A'
                if last_token == a_token_id:
                    # print(f"Debug: Penalizing ':' token (ID: {colon_token_id})") # Optional Debug
                    next_logits[colon_token_id] -= penalty_value

                # 3. *** ADDED: Penalize 'Q' if the last token was 'Q' ***
                if last_token == q_token_id:
                    # print(f"Debug: Penalizing 'Q' token (ID: {q_token_id})") # Optional Debug
                    next_logits[q_token_id] -= penalty_value

            # Ensure logits are finite after penalties if not using inf
            # This helps prevent NaNs in softmax if a large finite penalty is used
            if penalty_value != float('inf'):
                 next_logits = torch.nan_to_num(next_logits, nan=-float('inf'))


            # --- Sampling ---
            try:
                if top_p is None:
                    # Greedy decoding
                    chosen_token = torch.argmax(next_logits).item()
                elif top_p >= 1.0: # Treat top_p=1.0 or >1.0 as full sampling
                    # Full sampling (softmax with temperature and penalties applied)
                    probs = F.softmax(next_logits, dim=-1)
                    # Handle potential NaNs in probs if all logits became -inf
                    if torch.isnan(probs).all():
                        print("Warning: All probabilities are NaN after penalties/softmax. Falling back to greedy.")
                        chosen_token = torch.argmax(next_logits).item() # Use argmax on penalized logits
                    else:
                        chosen_token = torch.multinomial(probs, 1).item()
                else: # top_p is < 1.0
                    # Top-p (nucleus) sampling (pass the already modified logits)
                    chosen_token = nucleus_sampling(next_logits, p=top_p)
            except Exception as e:
                 print(f"Error during token sampling at step {step_i}: {e}")
                 print(f"Logits shape: {next_logits.shape}, top_p: {top_p}")
                 # Fallback to argmax or stop
                 chosen_token = torch.argmax(next_logits).item()
                 print(f"Falling back to greedy token: {chosen_token}")
                 # break # Optionally stop generation on sampling error


            # --- Update Context ---
            context_tokens.append(chosen_token)

            # --- Monosemantic Analysis (Optional) ---
            neighbors = []
            if do_monosemantic and monosemantic_info is not None:
                # NOTE: Ensure monosemantic_analysis_for_token function exists and is compatible
                try:
                    # Replace with your actual function call if you have it
                    # neighbors = monosemantic_analysis_for_token(
                    #        chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                    # )
                    pass # Remove pass if you have the function call above
                except NameError:
                     pass # Silently ignore if function not available
                except Exception as e:
                     print(f"Error during monosemantic analysis: {e}") # Log other errors

            generated_token_data.append((chosen_token, neighbors))

    model.train(was_training) # Restore original training/eval mode

    # Decode the full generated sequence
    try:
        final_text = enc.decode(context_tokens)
    except Exception as e:
        print(f"Error decoding final text: {e}")
        final_text = "Error: Decoding failed."


    # --- Annotation Formatting (if needed) ---
    annotated_text = "Annotation formatting disabled by default." # Placeholder
    if do_monosemantic: # Only format annotations if requested
        try:
            prefix_text = enc.decode(enc.encode(init_text)) # Re-encode/decode ensures consistency
            annotated_strs = [prefix_text]

            # Add annotations to each *generated* token
            for (tid, neighs) in generated_token_data:
                token_str = enc.decode([tid])
                if neighs: # Only add annotation if analysis results exist
                    try:
                        # Adjust format based on what 'neighbors' actually contains
                        neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs] # Assumes format [(score, token_id), ...]
                        annotated = f"{token_str}[NN={neighbor_strs}]"
                    except (IndexError, TypeError, Exception) as e_inner:
                         # print(f"Warning: Could not format neighbors for token {tid}: {e_inner}") # Debug
                         annotated = f"{token_str}[NN=Error]" # Handle unexpected neighbor format
                else:
                    annotated = token_str
                annotated_strs.append(annotated)

            annotated_text = "".join(annotated_strs)
        except Exception as e:
             print(f"Error during annotation formatting: {e}")
             annotated_text = "Error: Annotation formatting failed."


    return final_text, annotated_text

################################################################################
# 8. Training (Updated: Removed penalty from loss calculation)
################################################################################

def compute_loss(logits, tokens):
    """
    Compute standard cross-entropy loss.
    Assumes logits are for predictions and tokens are ground truth.
    """
    seq_len, batch_size, vocab_size = logits.shape

    # Need at least 2 tokens to have a target
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Logits for predicting token t are at index t-1
    # Targets for token t are at index t
    preds = logits[:-1, :, :].reshape(-1, vocab_size) # Shape: ((seq_len-1)*batch_size, vocab_size)
    gold = tokens[1:, :].reshape(-1)                  # Shape: ((seq_len-1)*batch_size)

    # Ensure gold tokens are within vocab bounds (optional sanity check)
    # gold = torch.clamp(gold, 0, vocab_size - 1)

    loss = F.cross_entropy(preds, gold) # Ignore index if padding is used

    return loss


# --- Update train_one_model to use the new compute_loss ---
# --- and ensure 'enc' is passed for sampling ---

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    enc, # *** Added encoder here ***
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    monosemantic_info=None, # Keep for sampling
                    prompt="Once upon a"):
    """
    Train loop using standard cross-entropy loss.
    Generates samples using the updated generate_text with penalties applied internally.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0
        step_in_epoch = 0

        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1
            batch_tokens = batch_tokens.to(device) # (batch_size, seq_len)
            # Reshape batch_tokens for model if needed, e.g., (seq_len, batch_size)
            # Assuming model expects (seq_len, batch_size) based on original loss code:
            batch_tokens_model = batch_tokens.transpose(0, 1) # (seq_len, batch_size)

            logits = model(batch_tokens_model) # Forward pass -> (seq_len, batch_size, vocab_size)

            # Use the standard loss function (no penalties here)
            loss = compute_loss(logits, batch_tokens_model)

            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            partial_loss += loss_item
            partial_count += 1

            if batch_idx % log_steps == 0:
                 if partial_count > 0:
                     avg_part_loss = partial_loss / partial_count
                     print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                           f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                           f"Partial Avg Loss: {avg_part_loss:.4f}")
                     partial_loss = 0.0
                     partial_count = 0

            # Periodically generate sample text using the updated generate_text
            current_time = time.time()
            # Check enc is not None before sampling
            if enc is not None and current_time >= next_sample_time :
                with torch.no_grad():
                    # Ensure model is in eval mode for generation inside training loop
                    model.eval()
                    print(f"\n[{model_name}] Generating sample text (penalties ON during generation) at epoch={epoch}, step={batch_idx}...")

                    # Generate using different sampling methods - penalties applied inside generate_text
                    for p_val, p_name in [(None, "Greedy"), (0.95, "Top-p=0.95"), (1.0, "Full Sample (p=1.0)")]:
                        text_sample, ann_sample = generate_text(
                            model, enc, prompt, max_new_tokens=30, device=device,
                            top_p=p_val,
                            temperature=0.8, # Example temperature
                            monosemantic_info=monosemantic_info,
                            do_monosemantic=(monosemantic_info is not None),
                            apply_penalties=True # Penalties handled inside generate_text
                        )
                        print(f" {p_name} Sample: {text_sample}")
                        # print(f" Annotated: {ann_sample}") # Optional

                    model.train() # Set back to train mode

                next_sample_time = current_time + sample_interval

            # Optional early stopping within epoch
            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        # Print average loss for this epoch
        if step_in_epoch > 0:
            avg_loss = total_loss / step_in_epoch
            print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}\n")
        else:
            print(f"[{model_name}] *** End of Epoch {epoch} *** No steps completed.")

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
        # "lstm_seq": lstm_model,
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

        with torch.no_grad():
            temperatures = [0.7, 1.0, 1.5]
            top_ps = [None, 0.95, 1.0]  # None = greedy, 0.95 = nucleus, 1.0 = full distribution

            for top_p in top_ps:
                for temp in temperatures:
                    tag = (
                        "greedy" if top_p is None else
                        f"top-p={top_p}"
                    )
                    print(f"[{model_name}] Sample ({tag}, temperature={temp}) from prompt: '{args.prompt}'")

                    text, ann = generate_text(
                        model, enc, args.prompt,
                        max_new_tokens=100,
                        device=device,
                        top_p=top_p,
                        temperature=temp,
                    )

                    print(text)
                    print(f"Annotated:\n{ann}\n")

            print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()