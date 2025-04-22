import argparse
from pathlib import Path
import pandas as pd


def process_file(in_path: Path, out_path: Path):
    df = pd.read_parquet(in_path, engine="pyarrow")
    triviadf = pd.concat(
        [df["question"], df["answer"]], axis=1, keys=["question", "answer"]
    )
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in triviadf.iterrows():
            f.write(f"Q: {row['question']} A: {row['answer']['value']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parse all 'train' parquet files in a folder to Q/A text files."
    )
    parser.add_argument(
        "dirpath", type=Path, help="Path to folder containing parquet files"
    )
    args = parser.parse_args()

    if not args.dirpath.is_dir():
        raise SystemExit(f"{args.dirpath} is not a folder")

    files = sorted(args.dirpath.glob("*train*.parquet"))
    if not files:
        print("No files matching '*train*.parquet' found.")
        return

    for idx, parquet_file in enumerate(files, start=1):
        out_file = args.dirpath / f"parsedTriviafile_{idx}.txt"
        print(f"Processing {parquet_file.name} → {out_file.name}")
        process_file(parquet_file, out_file)


if __name__ == "__main__":
    main()
