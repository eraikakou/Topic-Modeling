import pandas as pd
import spacy
from nltk.corpus import stopwords
from lda_sklearn import LdaSklearn
from src.lda_gensim import LdaGensim
from src.utils import text_preprocessing


if __name__ == '__main__':
    # Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
    # Run in terminal: python -m spacy download en
    nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])

    df = pd.read_csv("../data/ubs-mobile-app-reviews.csv")
    df = df.dropna(subset=["content"])

    #df["clean_content"] = df["content"].apply(lambda x: text_preprocessing.preprocess_text(x))
    #df.to_csv("../data/ubs-mobile-app-reviews-clean.csv")

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    data = df.content.values.tolist()

    lda_sklearn = LdaSklearn(nlp)
    lda_sklearn.run(df, 5)
    # lda_gensim = LdaGensim()
    # lda_gensim.run(df, 4, stop_words, nlp)
