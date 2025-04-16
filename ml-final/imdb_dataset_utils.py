import pandas as pd
import datasets

def load_imdb_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)

    inputs = []
    outputs = []

    for _, row in df.iterrows():
        overview = row["Overview"].strip()
        title = row["Series_Title"].strip()
        cast = ", ".join([
            row["Star1"].strip(),
            row["Star2"].strip(),
            row["Star3"].strip(),
            row["Star4"].strip()
        ])
        genre = row["Genre"].strip()
        rating = str(row["IMDB_Rating"]).strip()

        prompt = f"Overview: {overview}\nGenre: {genre}\nRating: {rating}"
        completion = f"Title: {title}\nCast: {cast}"

        inputs.append(prompt)
        outputs.append(completion)

    return datasets.Dataset.from_dict({"input": inputs, "output": outputs})