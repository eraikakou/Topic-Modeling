import pandas as pd
from src.utils import text_preprocessing


if __name__ == '__main__':
    df = pd.read_csv("../data/ubs-mobile-app-reviews.csv")

    df = df[
        ~(df["content"].isnull())
    ]

    df["clean_content"] = df["content"].apply(lambda x: text_preprocessing.preprocess_text(x))
    df.to_csv("../data/ubs-mobile-app-reviews-clean.csv")
