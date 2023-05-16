import numpy as np
import pandas as pd
import pickle
import nltk
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
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
"""
X_train, X_test, y_train, y_test = train_test_split(embedding_train, labels, test_size=0.2, random_state=42)

clf = CatBoostClassifier(
        iterations=1000,
        random_seed=42,
#        task_type='GPU',
#         text_features=['text'],
    )
clf.fit(X_train, y_train, eval_set=(X_train, y_train), verbose=10, use_best_model=True, plot=False,)

predictions = clf.predict(X_test)
"""
df_train = pd.DataFrame(embedding_train)
df_train['target'] = labels
df_train['text'] = processed_features


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

history_acc = []
history_predictions = []
clfs = []
for train_index, val_index in kf.split(df_train.drop(['target', 'text'], axis=1), df_train['target']):
    train_df, val_df = df_train.iloc[train_index].reset_index(drop=True), df_train.iloc[val_index].reset_index(drop=True)
    y_train, y_val = train_df['target'], val_df['target']
    X_train, X_val = train_df.drop(['target', 'text'], axis=1), val_df.drop(['target', 'text'], axis=1)
    clf = CatBoostClassifier(
        iterations=100,
        random_seed=42,
        task_type='GPU',
#         text_features=['text'],
    )
    clf.fit(X_train, y_train, eval_set=(X_val, y_val),
            verbose = 10, use_best_model = True, plot = False,)
    clfs.append(clf)
    predictions = clf.predict(X_val)
    history_predictions.append(predictions)
    history_acc.append(accuracy_score(y_val, predictions))
print(history_acc, np.mean(history_acc))
#print("total acc", accuracy_score(y_val, predictions))


data1 = pd.read_csv('./dictionary_documents/positive.csv', delimiter=';', header=None)
data1 = data1[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data1=data1[-10000:]
data2 = pd.read_csv('./dictionary_documents/negative.csv', delimiter=';', header=None)
data2 = data2[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data2=data2[-10000:]
data = pd.concat([data1, data2])
data = data[['x', 'y']]

features_test = data['x'].values
labels_test = data['y'].astype('int').values

#preprocessing data
processed_features = []
for sentence in range(0, len(features_test)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features_test[sentence]))

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
embedding_test = model.encode(processed_features)
print(processed_features)
total_predict = pd.DataFrame()
for i, clf in enumerate(clfs):
    total_predict[i] = clf.predict(embedding_test).astype(int)
print(total_predict)
total_predict["total"] = total_predict[[0, 1, 2,]].mode(axis=1)[0]
total_predict["total"] = total_predict["total"].astype(int)
print("total_acc", accuracy_score(total_predict["total"], labels_test ))
