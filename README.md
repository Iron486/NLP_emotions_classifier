# NLP_emotions_classifier
Sentiment analysis on emotions dataset

The aim of this problem was the correct classification of emotions of this dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp.



The train, validation and test datasets with both sentences and emotion labels (sadness,anger,love,surprise,fear,joy) were provided.

In this repository there are the following notebooks:

- [Preprocessing_EDA_LSTM.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Preprocessing_EDA_LSTM.ipynb) is the notebook that I used to preprocess the data, apply exploratory data analysis on datasets and LSTM. A 100 GloVe encoding dimension vector developed by [Stanford University](https://nlp.stanford.edu/projects/glove/) was used to embed each word in the dataset.
- [EDA_LSTM_50_encodings.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/EDA_LSTM_50_encodings.ipynb) and [EDA_LSTM_200_encodings.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/EDA_LSTM_200_encodings.ipynb) that are like the previous one, but using 50 and 200 dimension encoding vectors.
- [LSTM_Conv1d.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/LSTM_Conv1d.ipynb) where I applied an LSTM layer with a 1d convolutional layer.
- [Pretrained_BERT.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Pretrained_BERT.ipynb) in which I applied BERT.
- [Pretrained_Bert_stopword_lemmatizer.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Pretrained_Bert_stopword_lemmatizer.ipynb)  in which I applied stopword and lemmatizer, followed by BERT. It was the model with the highest accuracy.
- [Sentiment_prediction.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Sentiment_prediction.ipynb) is a class prediction notebook based on a single sentence that the user gives as input.

### PREPROCESSING

The embeddings where downloaded from here https://nlp.stanford.edu/projects/glove/ and they were transformed into a dictionary and saved so that it required less time to load them. The same thing has been done with embeddings of 50 and 200 length-vectors.
The datasets were loaded too and words were tokenized and padded to a fixed maximum length.
Also, labels were encoded based on the train data with the following encoder from keras `LabelEncoder()` and GloVe weights were added based on the words present in the train dataset.

For BERT models, the words were tokenized with the `AutoTokenizer` class from `transformers` library, using the `from_pretrained()` method and using `bert-base-cased` as argument, since the BERT has . 
Instead, for the LSTM and the other tested variations `Tokenizer()` method from `keras.preprocessing.text` was used.


### EXPLORATORY DATA ANALYSIS

For this senmtiment analysis problem, 2 types of graphs were plotted.

The first one depicts a wordcloud graph, imported using the library called `wordcloud`.


<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/169670215-b074d597-5abf-4944-bce9-054e5decdf65.png" width="500" height="500"/>   </p>

<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/169670221-2b401cc2-7f53-4353-9eeb-814a13dc923c.png" width="500" height="500"/>   </p>

<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/169670207-6e737fe6-05c8-4919-be15-8f96f75662d9.png" width="500" height="500"/>   </p>

It's clear that the words 'feel' and 'feeling' are the most common words for all the three datasets. This is due to the fact that these are the main verbs used to describe all the types of feelings present in the 6 classes.

Below, I put the code used to plot the 3 graphs:

bert=TFBertModel.from_pretrained('bert-base-cased') 
```python
def plot_cloud(wordcloud,intt,dataset):
    axes[intt].set_title('Word Cloud '+dataset+' dataset', size = 19,y=1.04)
    axes[intt].imshow(wordcloud)             
    axes[intt].axis("off"); # No axis details
from wordcloud import WordCloud
fig, axes = plt.subplots(3,1, figsize=(25, 41), sharey=True)
wordcloud = WordCloud(width = 600, height = 600,background_color = 'White',max_words=1000,repeat=False,min_font_size=5,collocations=False).generate(b_train)           #collocation: gets rid of the repeated words                                          

plot_cloud(wordcloud,0,'train')

wordcloud = WordCloud(width = 600, height = 600,background_color = 'White',max_words=1000,repeat=False,min_font_size=5,collocations=False).generate(b_val)
plot_cloud(wordcloud,1,'validation')

wordcloud = WordCloud(width = 600, height = 600,background_color = 'White',max_words=1000,repeat=False,min_font_size=5,collocations=False).generate(b_test)
plot_cloud(wordcloud,2,'test')
```
