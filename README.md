# pico-llm

<!-- Let's write our answers to the questions here -->

## Q1

Sanity check that you are able to run the code, which by default will only run an LSTM on TinyStories. It is possible that the code is too slow or runs out of memory for you: consider using an aggressive memory saving command-line argument such as “--block_size 32", and also using the simplified sequence data via “--tinystories weight 0.0 --input_files 3seqs.txt --prompt "0 1 2 3 4"”. Make sure you understand the code, in particular the routine torch.nn.Embedding, which has not been discussed in class; why is that routine useful?

The input data consists of token IDs, which are integers from 0 to vocab_size -1. Neural networks don't understand integers as categorical identities. Feeding raw token IDs would make the model assume numeric relationships between words (e.g. token 5 is closer to 6 than 500), which is meaningless. So instead, each token is mapped to a learned embedding vector (e.g. 128 or 512-dimensional), which can capture semantic properties and relationships between tokens.

When a batch of token sequences comes in (shape: (seq_len, batch)), the embedding layer converts them into shape (seq_len, batch, embed_size) — a form that the neural network can understand and learn from.
During training, these embeddings are updated via backpropagation just like any other model parameter, so they learn to represent useful features of language.

## Possible Theory About Problems with K-Gram MLP

When given `python pico.py --device_id cuda --input_files input_files/input.txt --prompt "hello" --block_size 32 --tinystories_weight 0.5` using new K-Gram MLP code I found that it started running much more steps in each Epoch. Roughly Epochs were of size 1250 steps and the step count appeared to print every 100 steps or so with 3 total Epochs. What was happening in our old code was that at the end of every Epoch the avg loss was not always decreasing aznd I think this can be explained a bit by what I was seeing when I did the large run. When I did the large run within each Epoch I noticed the avg loss was jumping up and down slightly (not constantly decreasing), but by the end of the Epoch the average was always less than the previous Epoch. What I think happened was during our old method our sample size for Epochs was only 1 step meaning that there was not enough points on the average for it to decrease properly so we were constantly left with the rare jump instance where the current loss was greater than the previous.

## Q2

Does it make sense for you to use torch.nn.Embedding?

Even though your model is an MLP operating on k-grams (sliding windows of k tokens), those tokens still start as integers (token IDs) — and as discussed earlier, neural networks can’t meaningfully process raw integers.

By embedding each token before feeding the window to the MLP, you get: A dense vector representation per token (shape: [embed_size]), and a k × embed_size vector for each sliding window, formed by concatenating the k embeddings. This forms the MLP input: a vector that summarizes a local context of k tokens.
torch embeddings turns discrete token IDs into dense vectors for processing.

## Q3

Does the generated text change in interesting ways with nucleus_sampling?

Nucleus sampling chooses the next token randomly, but only from the smallest possible set of tokens whose total probability mass exceeds p (like 95%).

This allows: balance between coherence and diversity, avoiding degenerate or repetitive outputs, allowing creative or unexpected continuations.
It allows your model to show off more of what it’s learned by breaking out of the most-likely path and exploring creative possibilities. It’s especially useful for generative tasks like storytelling or dialogue.