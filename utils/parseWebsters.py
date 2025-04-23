import json
import argparse

# Script for parsing websters english dictionary data (cloned from https://github.com/matthewreagan/WebstersEnglishDictionary)
# into a Q/A text format 
# for training. Run from root directory as
# `python parsewebsters.py training_data/SQuAD/train-00000-of-00001.parquet`

parser = argparse.ArgumentParser()
parser.add_argument("filepath")
args = parser.parse_args()

file = open(args.filepath)

data = json.load(file)

f = open('parsedDictionary.txt','w',encoding="utf-8")
for key, value in data.items():
    f.write(f'Q: What does {key} mean? A: {value} \n')
    
file.close()
f.close()
