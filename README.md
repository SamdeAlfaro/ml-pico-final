# pico-llm

<!-- Let's write our answers to the questions here -->

## Q1

Sanity check that you are able to run the code, which by default will only run an LSTM on TinyStories. It is possible that the code is too slow or runs out of memory for you: consider using an aggressive memorysaving command-line argument such as “--block data via “--tinystories weight 0.0--input size 32”, and also using the simplified sequence files 3seqs.txt--prompt "0 1 2 3 4"”. Make sure you understand the code, in particular the routine torch.nn.Embedding, which has not been discussed in class; why is that routine useful?

The torch.nn.Embedding routine is used to transform the words/tokens into dense vectors of fixed size. This is useful in NLP tasks--the words/tokens have semantic meanings, so similar words with similar meaning will have similr vector representations. This will help the network learn relationships better.
The embedding layer is initialized with random values and is trained during the training phase.

# TO DO:
Keep answering questions here...