## This project was implemented as part of ITMO University's,  Natural Language Processing course (stream 4, spring 2023) on ods.ai

### Describe
The idea of the project is to use trained models for classifying the emotional coloring of the text to assess the concentration of negative messages in business chats. Because it is a very big problem to make sure that there are no conflicts and violations in the chat. In order not to administer the chat all the time, we suggest using a chatbot that will send notifications to the administrator only in case the concentration of negative messages is too high. The rest of the time, you don't have to follow the chat at all

### Dataset
We used a dataset with messages that are most suitable for the task of classifying short message texts in the article at the link:
  http://www.swsys.ru/index.php?page=article&id=3962&lang=
Unfortunately, today I have not found the dataset from this article in the public domain, so I have some doubts about the legality of its use. 
In this dataset there are 114911 positive and 111923 negative messages with various additional data from the social network Twitter posts
Since we don't have a dataset with text messages in the chat, the best we could afford was Twitter posts. It seems to me that although they are not chat messages, they are all the closest to them. To date, we have nothing better at our disposal.

### Chat-bot
We quickly sketched out a script for a Telegram chatbot to check how realistic it is to launch such a bot. No obvious problems were found, with a stable Internet, the chatbot gives instant notifications.
### Train
As of 21.05.23 you can observe 4 versions of notebooks with training of various models and collection of their metrics
### Requirements
numpy 1.24.3  
pandas 2.0.1  
nltk 3.8.1  
sentence-transformers 2.2.2  
sklearn 1.2.2  
catboost 1.2  
torch 2.0.1  
transformers 4.29.2  

## Team

- [Mikhaylov Alexey](https://t.me/sp1derAlex) 
- [Baranov Vitaly](https://t.me/vitalybar)