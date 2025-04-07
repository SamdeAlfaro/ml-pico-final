# pico-llm

<!-- Let's write our answers to the questions here -->

## Q1

Sanity check that you are able to run the code, which by default will only run an LSTM on TinyStories. It is possible that the code is too slow or runs out of memory for you: consider using an aggressive memorysaving command-line argument such as “--block data via “--tinystories weight 0.0--input size 32”, and also using the simplified sequence files 3seqs.txt--prompt "0 1 2 3 4"”. Make sure you understand the code, in particular the routine torch.nn.Embedding, which has not been discussed in class; why is that routine useful?

The torch.nn.Embedding routine is used to transform the words/tokens into dense vectors of fixed size. This is useful in NLP tasks--the words/tokens have semantic meanings, so similar words with similar meaning will have similr vector representations. This will help the network learn relationships better.
The embedding layer is initialized with random values and is trained during the training phase.

# TO DO:
Keep answering questions here...

# Possible Theory About Problems with K-Gram MLP 
When given `python pico.py --device_id cuda --input_files input_files/input.txt --prompt "hello" --block_size 32 --tinystories_weight 0.5` using new K-Gram MLP code I found that it started running much more steps in each Epoch. Roughly Epochs were of size 1250 steps and the step count appeared to print every 100 steps or so with 3 total Epochs. What was happening in our old code was that at the end of every Epoch the avg loss was not always decreasing aznd I think this can be explained a bit by what I was seeing when I did the large run. When I did the large run within each Epoch I noticed the avg loss was jumping up and down slightly (not constantly decreasing), but by the end of the Epoch the average was always less than the previous Epoch. What I think happened was during our old method our sample size for Epochs was only 1 step meaning that there was not enough points on the average for it to decrease properly so we were constantly left with the rare jump instance where the current loss was greater than the previous.