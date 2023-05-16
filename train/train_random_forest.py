import numpy as np
import pandas as pd
import pickle
import nltk
import re

# get and prepare data to preprocessing
data1 = pd.read_csv('./dictionary_documents/positive.csv', delimiter=';', header=None, nrows=60000)
data1 = data1[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data2 = pd.read_csv('./dictionary_documents/negative.csv', delimiter=';', header=None, nrows=60000)
data2 = data2[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data = pd.concat([data1,data2])
data = data[['x','y']]





# get features and labels
features = data['x'].values
labels = data['y'].astype('int').values

#preprocessing data
processed_features = []
for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[а-яА-Я]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[а-яА-Я]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
# download vectorizer


vectorizer = TfidfVectorizer (max_features=3000, min_df=0.0001, max_df=0.8, stop_words=stopwords.words('russian'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

pickle.dump(vectorizer.vocabulary_, open("feature.pkl", "wb"))

from sklearn.model_selection import train_test_split
# split data to train
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestClassifier
# download classifier
text_classifier = RandomForestClassifier(n_estimators=250, random_state=0)
# train
text_classifier.fit(X_train, y_train)

# eval metrics
predictions = text_classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))
import pickle
# save model
with open('rf_model_1', 'wb') as f:
    pickle.dump(text_classifier, f)