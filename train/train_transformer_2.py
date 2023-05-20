import numpy as np
import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, DistilBertTokenizer

from model_transformer import DistilBertForClassification
print(torch.cuda.is_available())
# Загрузка датасета

data1 = pd.read_csv('./dictionary_documents/positive.csv', delimiter=';', header=None, nrows=10)
data1 = data1[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data2 = pd.read_csv('./dictionary_documents/negative.csv', delimiter=';', header=None, nrows=10)
data2 = data2[[3, 4]].rename({3: "x", 4: "y"}, axis='columns')
data = pd.concat([data1, data2])
data = data[['x', 'y']]
data.loc[data['y'] < 0, 'y'] = 0

features_test = data['x'].values
labels_test = data['y'].astype('int').values

# preprocessing data
processed_features = []
for sentence in range(0, len(features_test)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features_test[sentence]))

    # remove all single characters
    processed_feature = re.sub(r'\s+[а-яА-Я]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[а-яА-Я]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)
# print(labels_test)
# Разделение на тренировочную и тестовую выборки
train_texts, val_texts, train_labels, val_labels = train_test_split(processed_features, labels_test, test_size=0.2)

# Инициализация токенизатора и модели BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased")
config = {
    "num_classes": 2,
    "dropout_rate": 0.1,

}
model = BertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)


# Создание класса для датасета
class MovieReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_seq_len,
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Создание объектов для тренировочной и тестовой выборок
train_dataset = MovieReviewDataset(train_texts, train_labels, tokenizer, 128)
val_dataset = MovieReviewDataset(val_texts, val_labels, tokenizer, 128)

# Создание DataLoader'ов для пакетного обучения модели
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Определение параметров обучения
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
num_epochs = 4

# Обучение модели
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch}, Training Loss: {train_loss / len(train_loader)}')

# Оценка модели на тестовой выборке
model.eval()
with torch.no_grad():
    val_preds = []
    for batch in val_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        val_preds.extend(preds.cpu().detach().numpy().tolist())

# Оценка качества модели
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

val_accuracy = accuracy_score(val_labels, val_preds)
val_precision, val_recall, val_fscore, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
print(f"Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}")
