# Transformer
- Structure of our transformer model consists of an embedding layer derived from our input sequences, transformer blocks with self attention and feedforward layers
  - https://huggingface.co/blog/not-lain/tensor-dims <- fairly close to our architecture
  - We take the tokens and create an embedding layer, Assemble a number of Transformer blocks with the self-attention(normalization and multi-head self attention) and feedforward layers(where we run an mlp)
  - Output is normalized a final time