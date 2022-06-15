import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from pprint import pprint
from src.utils import topic_modeling


class LdaGensim:

    def __init__(self):

        self.lda_model = None
        self.bigram_mod = None
        self.trigram_mod = None

    def create_bigrams_trigrams_model(self, data):
        """
        Bigrams are 2 words frequently occuring together in docuent. Trigrams are 3 words frequently occuring.
        The 2 arguments for Phrases are min_count and threshold. The higher the values of these parameters ,
        the harder its for a word to be combined to bigram.
        """

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data], threshold=100)
        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(trigram)

    def make_bigrams(self, texts):
        return [self.bigram_mod[doc] for doc in texts]

    def make_trigrams(self, texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    def measure_model_performance(self, data, corpus, id2word):
        """
        A model with higher log-likelihood and lower perplexity (exp(-1. * log-likelihood per word)) is considered to be good.
        - Lower the perplexity better the model.
        - Higher the topic coherence, the topic is more human interpretable.
        """
        # Compute Perplexity
        print('\nPerplexity: ', self.lda_model.log_perplexity(corpus))
        # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=data, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

    def run(self, df, num_of_topics, stop_words, nlp):
        data = df.content.values.tolist()

        # Tokenization
        data_words = list(topic_modeling.convert_sentence_to_words(data))

        self.create_bigrams_trigrams_model(data_words)

        # Remove Stop Words
        data_words = topic_modeling.remove_stopwords(data_words, stop_words)

        # Form Bigrams
        data_words = self.make_bigrams(data_words)

        data_lemmatized = topic_modeling.lemmatization_for_gensim(data_words, nlp)

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)
        # Create Corpus
        texts = data_lemmatized
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_of_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

        # Print the keyword of topics
        pprint(self.lda_model.print_topics())

        self.measure_model_performance(data_lemmatized, corpus, id2word)
        self.get_pyLDAvis(corpus, id2word)

    def get_pyLDAvis(self, corpus, id2word):
        vis = gensimvis.prepare(self.lda_model, corpus, id2word)
        pyLDAvis.save_html(vis, '../images/lda_gensim_results.html')
