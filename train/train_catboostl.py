import numpy as np
import pandas as pd
import pickle
import nltk
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier

# get and prepare data to preprocessing
data1 = pd.read_csv('./dictionary_documents/positive.csv', delimiter=';', header=None, nrows=100000)
data1 = data1[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data2 = pd.read_csv('./dictionary_documents/negative.csv', delimiter=';', header=None, nrows=100000)
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

    # remove all non russian words
    processed_feature= re.sub("[^А-Яа-я]", " ", processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[а-яА-Я]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_train = model.encode(processed_features)

X_train, X_test, y_train, y_test = train_test_split(embedding_train, labels, test_size=0.2, random_state=42)

clf = CatBoostClassifier(
        iterations=1000,
        random_seed=42,
#        task_type='GPU',
#         text_features=['text'],
    )
clf.fit(X_train, y_train, eval_set=(X_train, y_train), verbose=10, use_best_model=True, plot=False,)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))