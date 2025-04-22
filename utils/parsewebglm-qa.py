import json
import argparse
import pandas as pd    

# Script for parsing webglm-qa data (cloned from https://huggingface.co/datasets/THUDM/webglm-qa)
# into a Q/A text format 
# for training. Run from root directory as
# `python parsewebglm-qa.py training_data/SQuAD/train-00000-of-00001.parquet`

parser = argparse.ArgumentParser()
parser.add_argument("filepath")
args = parser.parse_args()

jsonObj = pd.read_json(path_or_buf=args.filepath, lines=True)

f = open('parsedWebGlm.txt','w',encoding="utf-8")
for col,row in jsonObj.iterrows():
    f.write(f'Q: {row['question']} A: {row['answer']} \n')
