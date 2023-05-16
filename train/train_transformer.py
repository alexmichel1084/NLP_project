import os
import pandas as pd
import transformers
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, DistilBertModel
import re

from model_transformer import *
from trainer import Trainer

PATH = ""
MAX_LEN = 128
BATCH_SIZE = 64

# get and prepare data to preprocessing
data1 = pd.read_csv('./dictionary_documents/positive.csv', delimiter=';', header=None, nrows=60000)
data1 = data1[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data2 = pd.read_csv('./dictionary_documents/negative.csv', delimiter=';', header=None, nrows=60000)
data2 = data2[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data = pd.concat([data1,data2])
data = data[['x','y']]

train_split, val_split = train_test_split(data, train_frac=0.85)

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", truncation=True, do_lower_case=True)
"""
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')

train_dataset = RottenTomatoesDataset(train_split, tokenizer, MAX_LEN)
val_dataset = RottenTomatoesDataset(val_split, tokenizer, MAX_LEN)
test_dataset = RottenTomatoesDataset(test_data, tokenizer, MAX_LEN)
"""
train_params = {"batch_size": BATCH_SIZE,
                "shuffle": True,
                "num_workers": 0
                }

test_params = {"batch_size": BATCH_SIZE,
               "shuffle": False,
               "num_workers": 0
               }
# get features and labels
train_features = train_split['x'].values
train_labels = train_split['y'].astype('int').values

val_features = val_split['x'].values
val_labels = val_split['y'].astype('int').values
"""
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
    """
train_dataloader = DataLoader(train_split, **train_params)
val_dataloader = DataLoader(val_split, **test_params)
#test_dataloader = DataLoader(test_dataset, **test_params)

config = {
    "num_classes": 6,
    "dropout_rate": 0.05,

}
#model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base', num_labels=6, ignore_mismatched_sizes=True)

model = DistilBertForClassification(
    "distilbert-base-uncased",
    config=config
)

trainer_config = {
    "lr": 3e-5,
    "n_epochs": 1,
    "weight_decay": 1e-7,
    "batch_size": BATCH_SIZE,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
t = Trainer(trainer_config)
t.fit(
    model,
    train_dataloader,
    val_dataloader
)

"""
t.save("baseline_model.ckpt")

t = Trainer.load("baseline_model.ckpt")
"""
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

predictions = t.predict(features_test)
