import pandas as pd
import pyarrow as pa
import argparse

# Simple script for parsing a parquet file into a Q/A text format 
# for training. Run from root directory as
# `python parseSQuADFile.py training_data/SQuAD/train-00000-of-00001.parquet`

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="parquet file path")
args = parser.parse_args()

df = pd.read_parquet(args.filepath, engine="pyarrow")

QAdf = pd.concat([df['question'],df['answers']],axis=1,keys=['question','answers'])    

f = open('parsedSQuADfile.txt','w',encoding="utf-8")

for index, row in QAdf.iterrows():
    f.write(f'Q: {row['question']} A: {row['answers']['text'][0]} \n')

f.close()