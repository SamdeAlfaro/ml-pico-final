These utility scripts are inteded for parsing various 
datasets into a usable format for our model. 

Instructions for running each script can be found in the script itself.
The scripts assume that you have a local folder 'raw_training_data'
to which you are cloning your data sets via git from hugging face. 
Be sure to install
pyarrow and pandas for processing parquet files.

Training data for each script are sourced on hugging face 
at the following links:

[SQuAD](https://huggingface.co/datasets/rajpurkar/squad)
[TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
[webglm-qa](https://huggingface.co/datasets/THUDM/webglm-qa)
