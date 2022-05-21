# NLP_emotions_classifier
Sentiment analysis on emotions dataset

The aim of this problem was the correct classification of emotions of this dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp.

The train, validation and test datasets with both sentences and emotion labels were provided.

In this repository there are the following notebooks:

- [Preprocessing_EDA_LSTM.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Preprocessing_EDA_LSTM.ipynb) is the notebook that I used to preprocess the data, apply exploratory data analysis on datasets and LSTM. A 100 GloVe encoding dimension vector developed by [Stanford University](https://nlp.stanford.edu/projects/glove/) was used to embed each word in the dataset.
- [EDA_LSTM_50_encodings.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/EDA_LSTM_50_encodings.ipynb) and [EDA_LSTM_200_encodings.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/EDA_LSTM_200_encodings.ipynb) that are like the previous one, but using 50 and 200 dimension encoding vectors.
- [LSTM_Conv1d.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/LSTM_Conv1d.ipynb) where I applied an LSTM layer with a 1d convolutional layer.
- [Pretrained_BERT.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Pretrained_BERT.ipynb) in which I applied BERT.
- [Pretrained_Bert_stopword_lemmatizer.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Pretrained_Bert_stopword_lemmatizer.ipynb)  in which I applied stopword and lemmatizer, followed by BERT. It was the model with the highest accuracy.
- [Sentiment_prediction.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Sentiment_prediction.ipynb) is a class prediction notebook based on a single sentence that the user gives as input.
