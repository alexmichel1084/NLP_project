import pickle
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
SENT_DETECTOR = nltk.data.load("tokenizers/punkt/english.pickle")


class Model:
    def __init__(self, model_path="rf_model", features_path="feature.pkl"):
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

        self.tfidf = TfidfVectorizer(
            max_features=2500,
            min_df=7,
            max_df=0.8,
            stop_words=stopwords.words("russian"),
            vocabulary=pickle.load(open(features_path, "rb")),
        )

    def predict(self, my_comment):
        return self.model.predict(self.tfidf.fit_transform(my_comment))
