import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src.utils import topic_modeling


class LdaSklearn:

    def __init__(self, nlp):
        self.vectorizer = CountVectorizer(
            analyzer="word",
            min_df=10,
            # consider words that has occurred at least 10 times
            stop_words="english",
            # remove stop words
            lowercase=True,
            # convert all words to lowercase
            token_pattern="[a-zA-Z0-9]{2,}",
            # A word can contain numbers and alphabets of at least length 2 in order to be qualified as a word.
            # max_features=50000,
            # max number of uniq words
        )
        self.lda_model = None
        self.nlp = nlp

    def create_document_word_matrix(self, data):
        """
        The LDA topic model algorithm requires a document word matrix as the main input.

        You can create one using CountVectorizer.

        """

        data_vectorized = self.vectorizer.fit_transform(data)

        return data_vectorized

    def determine_the_best_model(self, data):
        """Use GridSearch to determine the best LDA model.

        The most important tuning parameter for LDA models is n_components (number of topics).

        In addition, I am going to search learning_decay (which controls the learning rate) as well.
        Besides these, other possible search params could be learning_offset (downweigh early iterations. Should be > 1)
        and max_iter. These could be worth experimenting if you have enough computing resources.
        Be warned, the grid search constructs multiple LDA models for all possible combinations of param values in the
        param_grid dict. So, this process can consume a lot of time and resources.

        """
        # Define Search Param
        search_params = {"n_components": [4, 5, 7, 10], "learning_decay": [.5, .7, .9]}

        # Init the Model
        lda = LatentDirichletAllocation(max_iter=5, learning_method="online", learning_offset=50., random_state=0)

        # Init Grid Search Class
        model = GridSearchCV(lda, param_grid=search_params)
        # Do the Grid Search
        model.fit(data)

        # Best Model
        self.lda_model = model.best_estimator_
        # Model Parameters
        print("Best Model's Params: ", model.best_params_)
        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)
        # Perplexity
        print("Model Perplexity: ", self.lda_model.perplexity(data))

    def build_lda_model(self, num_of_topics, data):
        """
        Everything is ready to build a Latent Dirichlet Allocation (LDA) model.
        Let’s initialise one and call fit_transform() to build the LDA model.

        """
        # Build LDA Model
        self.lda_model = LatentDirichletAllocation(
            n_components=num_of_topics,  # Number of topics
            max_iter=10,
            # Max learning iterations
            learning_method="online",
            random_state=100,
            # Random state
            batch_size=128,
            # n docs in each learning iter
            evaluate_every=-1,
            # compute perplexity every n iters, default: Don't
            n_jobs=-1,
            # Use all available CPUs
        )

        lda_output = self.lda_model.fit_transform(data)

        print(self.lda_model)  # Model attributes

        return lda_output

    def measure_model_performance(self, data):
        """
        A model with higher log-likelihood and lower perplexity (exp(-1. * log-likelihood per word)) is considered to be good.

        """
        # Log Likelyhood: Higher the better
        print("Log Likelihood: ", self.lda_model.score(data))
        # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
        print("Perplexity: ", self.lda_model.perplexity(data))
        # See model parameters
        pprint(self.lda_model.get_params())

    def show_topics(self, n_words=20):
        """Show top N keywords for each topic

        """
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in self.lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))

        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]

        topic_labels = ["Topic" + str(i) for i in range(self.lda_model.n_components)]
        df_topic_keywords["Topics"] = topic_labels

        return df_topic_keywords

    def create_document_topic_matrix(self, data):
        """
        To classify a document as belonging to a particular topic, a logical approach is to see which topic has the
        highest contribution to that document and assign it. In the table below, I’ve greened out all major topics in a
        document and assigned the most dominant topic in its own column.

        """
        # Create Document — Topic Matrix
        lda_output = self.lda_model.transform(data)

        # column names
        topicnames = ["Topic" + str(i) for i in range(self.lda_model.n_components)]

        # index names
        docnames = ["Doc" + str(i) for i in range(data.shape[0])]

        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic["dominant_topic"] = dominant_topic

        return df_document_topic, lda_output

    def create_topic_keyword_matrix(self, vectorizer):
        """
        """
        topicnames = ["Topic" + str(i) for i in range(self.lda_model.n_components)]

        # Topic-Keyword Matrix
        df_topic_keywords = pd.DataFrame(self.lda_model.components_)
        # Assign Column and Index
        df_topic_keywords.columns = vectorizer.get_feature_names()
        df_topic_keywords.index = topicnames

    def predict_topic(self, text, df_topic_keywords):
        """Predicts Topics using LDA model

        Assuming that you have already built the topic model, you need to take the text through the same routine
        of transformations and before predicting the topic. For our case, the order of transformations is:

        sent_to_words() –> Stemming() –> vectorizer.transform() –> best_lda_model.transform()
        You need to apply these transformations in the same order. So to simplify it, let’s combine
        these steps into a predict_topic() function.
        """

        # Step 1: Clean with simple_preprocess
        tokenized = list(topic_modeling.convert_sentence_to_words(text))
        # Step 2: Lemmatize
        lemmatized = topic_modeling.lemmatization(tokenized, self.nlp)
        # Step 3: Vectorize transform
        vectorized = self.vectorizer.transform(lemmatized)

        # Step 4: LDA Transform
        topic_probability_scores = self.lda_model.transform(vectorized)

        topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()

        # Step 5: Infer Topic
        infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]

        # topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
        return infer_topic, topic, topic_probability_scores

    def apply_predict_topic(self, text, df_topic_keywords):
        """Predicts topics of the given dataset.
        """
        text = [text]
        infer_topic, topic, prob_scores = self.predict_topic(text, df_topic_keywords)
        return (infer_topic)

    @staticmethod
    def cluster_and_plot_documents_with_similar_topics(num_of_topics, lda_output):
        """Clusters documents that share similar topics and plot

        You can use k-means clustering on the document-topic probabilioty matrix, which is nothing but lda_output object.
        Since out best model has 15 clusters, I’ve set n_clusters=15 in KMeans().

        Alternately, you could avoid k-means and instead, assign the cluster as the topic column number
        with the highest probability score.

        We now have the cluster number. But we also need the X and Y columns to draw the plot.

        For the X and Y, you can use SVD on the lda_output object with n_components as 2.
        SVD ensures that these two columns captures the maximum possible amount of information from lda_output in the first 2 components.

        We have the X, Y and the cluster number for each document.

        Let’s plot the document along the two SVD decomposed components. The color of points represents the cluster number (in this case) or topic number.

        """
        clusters = KMeans(n_clusters=num_of_topics, random_state=100).fit_predict(lda_output)
        # Build the Singular Value Decomposition(SVD) model
        svd_model = TruncatedSVD(n_components=2)  # 2 components
        lda_output_svd = svd_model.fit_transform(lda_output)
        # X and Y axes of the plot using SVD decomposition
        x = lda_output_svd[:, 0]
        y = lda_output_svd[:, 1]

        # Plot
        plt.figure(figsize=(12, 12))
        plt.scatter(x, y, c=clusters)
        plt.xlabel('Component 2')
        plt.xlabel('Component 1')
        plt.title("Segregation of Topic Clusters", )
        plt.savefig("../images/clusters.png")

    def run(self, df, num_of_topics):

        data = df.content.values.tolist()

        # Tokenization
        data_words = list(topic_modeling.convert_sentence_to_words(data))

        data_lemmatized = topic_modeling.lemmatization(data_words, self.nlp)

        # Create the Document-Word matrix
        data_vectorized = self.create_document_word_matrix(data_lemmatized)

        # Build LDA Model
        #lda_output = self.build_lda_model(num_of_topics, data_vectorized)

        # Diagnose model performance with perplexity and log-likelihood
        #self.measure_model_performance(data_vectorized)

        # Use GridSearch to determine the best LDA model.
        self.determine_the_best_model(data_vectorized)

        # Get the Dominant topic per document
        df_document_topic, lda_output = self.create_document_topic_matrix(data_vectorized)
        # print(df_document_topic)

        # Apply Style
        # df_document_topics = df_document_topic.head(15).style.applymap(topic_modeling.color_green).applymap(topic_modeling.make_bold)
        # print(df_document_topics)

        # Get the top 15 keywords each topic:
        df_topic_keywords = self.show_topics(n_words=15)
        print(df_topic_keywords)

        # Predict topic for a new, unseen document
        # text = ["The app crashes when I login with fingerprint"]
        # infer_topic, topic, prob_scores = self.predict_topic(text, df_topic_keywords)
        # print(topic)
        # print(infer_topic)
        # print(prob_scores)

        df["Topic"] = df["content"].apply(
            lambda x: self.apply_predict_topic(text=x, df_topic_keywords=df_topic_keywords))
        print(df.head())

        # Cluster documents that share similar topics and plot
        self.cluster_and_plot_documents_with_similar_topics(self.lda_model.n_components, lda_output)

    @staticmethod
    def color_green(val):
        color = "green" if val > .1 else "black"
        return "color: {col}".format(col=color)

    @staticmethod
    def make_bold(val):
        weight = 700 if val > .1 else 400
        return "font-weight: {weight}".format(weight=weight)
