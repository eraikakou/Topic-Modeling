import gensim


def convert_sentence_to_words(sentences: list):
    """Tokenizes each sentence into a list of words, removing punctuations and unnecessary characters altogether.

    Tokenization is the act of breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols
    and other elements called tokens. Tokens can be individual words, phrases or even whole sentences.
    In the process of tokenization, some characters like punctuation marks are discarded.
    """
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations


def remove_stopwords(texts, stop_words):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(texts, nlp, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """The Spacy package we used here is better than PorterStemmer, Snowball.
    Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to
    the roots of words known as a lemma. The advantage of this is, we get to reduce the total number of unique words
    in the dictionary. As a result, the number of columns in the document-word matrix
    (created by CountVectorizer) will be denser with lesser columns.
     You can expect better topics to be generated in the end.


    :param texts:
    :param nlp:
    :param allowed_postags:
    :return:
    """

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        #texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ["-PRON-"] else "" for token in doc]))
    return texts_out


def lemmatization_for_gensim(texts, nlp, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        #texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        texts_out.append([token.lemma_ if token.lemma_ not in ["-PRON-"] else "" for token in doc])
    return texts_out
