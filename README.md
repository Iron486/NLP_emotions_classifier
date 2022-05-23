# NLP_emotions_classifier

The aim of this problem was the correct emotions classification of this dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp.



The train, validation and test datasets with both sentences and emotion labels (sadness,anger,love,surprise,fear,joy) were provided. 

In this repository there are the following notebooks:

- [Preprocessing_EDA_LSTM.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Preprocessing_EDA_LSTM.ipynb) is the notebook that I used to preprocess the data, apply exploratory data analysis on the datasets and that I trained with an LSTM layer. A 100 GloVe encoding dimension vector developed by [Stanford University](https://nlp.stanford.edu/projects/glove/) was used to encode the words in the datasets.
- [EDA_LSTM_50_encodings.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/EDA_LSTM_50_encodings.ipynb) and [EDA_LSTM_200_encodings.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/EDA_LSTM_200_encodings.ipynb) that are like the previous one, but using 50 and 200 dimension encoding vectors.
- [LSTM_Conv1d.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/LSTM_Conv1d.ipynb) where I applied an LSTM layer with 100 dimension encoding vectors.
- [LSTM_LSTM.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/LSTM_LSTM.ipynb) that is a test with two LSTM layers.
- [Bidirectional_LSTM_Conv1d.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/LSTM_LSTM.ipynb) where I used a Bidirectional LSTM with a CONV1d layer stacked above it.
- [Pretrained_BERT.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Pretrained_BERT.ipynb) in which I applied BERT.
- [Pretrained_Bert_stopword_lemmatizer.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Pretrained_Bert_stopword_lemmatizer.ipynb)  in which I applied stopword and lemmatization, followed by BERT. It was the model with the highest accuracy.
- [Sentiment_prediction.ipynb](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Sentiment_prediction.ipynb) is a class prediction notebook based on a single sentence that the user gives as input.

In all the notebooks used for training, I used on top of the layer a fully connected neural network with 6 neurons as output layer and a variable number of neurons, dropout and hidden layers.

I used the Google Colab GPU to train all the models except for the BERT with stopwords and lemmatizer, in which I used my [local GPU](https://github.com/Iron486/Iron486/blob/main/local_GPU.ipynb).

### PREPROCESSING

The embeddings where downloaded from here https://nlp.stanford.edu/projects/glove/ and then they were transformed into a dictionary and saved, so that it would have required less time to load them. The same thing has been done with embeddings of 50 and 200 length vectors.
The datasets were loaded too and the words were tokenized and padded to a fixed maximum length.
Also, labels were encoded based on the train data with the following encoder from keras `LabelEncoder()`, and GloVe weights were added based on the words present in the train dataset.

For BERT models, the words were tokenized with the `AutoTokenizer` class from `transformers` library, using the `from_pretrained()` method and using `bert-base-cased` as argument. This is because the input of the model expects 2 features ("input_ids" and "attention_mask") that can be obtained with the mentioned tokenizer. 

The input of the tokenizer was made by words that were lemmatized (with `WordNetLemmatizer()` class) and on which stopwords were applied. The lemmatizer and the stopwords were downloaded from the NLTK library.

Instead, for the LSTM and the other tested variations `Tokenizer()` class from `keras.preprocessing.text` was used.

### EXPLORATORY DATA ANALYSIS

For this senmtiment analysis problem, 2 types of graphs were plotted.

The first one depicts a wordcloud graph, imported using the library called `wordcloud`.

&nbsp; 



<img src="https://user-images.githubusercontent.com/62444785/169670215-b074d597-5abf-4944-bce9-054e5decdf65.png" width="410" height="440"/> <img src="https://user-images.githubusercontent.com/62444785/169670221-2b401cc2-7f53-4353-9eeb-814a13dc923c.png" width="410" height="440"/> 
   
<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/169670207-6e737fe6-05c8-4919-be15-8f96f75662d9.png" width="410" height=440"/>  </p> 

&nbsp;
    
It's clear that the words 'feel' and 'feeling' are the most common words for all the three datasets. This is due to the fact that these are the main verbs used to describe all the types of feelings present in the 6 classes.

Below, I put the code used to plot the 3 graphs:


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
The second type of plot that I coded was a barplot representing the number of sentences for each label.
    
![immagine_2022-05-22_003327046](https://user-images.githubusercontent.com/62444785/169671041-6710ae2a-f03b-4120-bd69-be736ca561ec.png)

So, the dataset is unbalanced with 'sadness' and 'joy' labels dominating over the others.
   
### TRAINING THE MODEL    
    
In the table below I summed up the model used, along with the Tokenizer, and the associated accuracy on validation dataset.

&nbsp;    
    
|             Model              | Tokenizer   |  # total params | # trainable params  |   Validation Accuracy    |
|--------------|--------------|-----------|------|-------|    
|  LSTM 100 encodings | Tokenizer()  | 1,478,086 | 59,586 | 84.55 |        
|  LSTM 200 encodings  | Tokenizer() | 3,041,058 |  204,058  | 83.80 |       
|  LSTM 50 encodings  | Tokenizer() | 753,836 |  44,586  |  80.90   |       
|  LSTM conv1D 100 encodings| Tokenizer()| 1,473,970  |   55,470   |  83.90  |
|  LSTM-LSTM 100 encodings| Tokenizer() |  1,498,046  |  79,546  | 84.00  |
|  Bert                  | AutoTokenizer.from_pretrained('bert-base-cased')| 108,420,460  | 108,420,460|  85.55 |
|  Bert stopword-lemmatizer  | AutoTokenizer.from_pretrained('bert-base-cased')| 108,420,460 | 108,420,460 | 93.75 |    
|  BidirectionalLSTM-Conv1d     | Tokenizer() |   1,500,257   |  81,757   |     81.60      |       
    
The largest accuracy was obtained on the BERT with stopword and lemmatization. 
The model was a pretrained model written by [Hugging Face](https://huggingface.co/bert-base-cased) , and I fetched it with the Tensorflow method `TFBertModel.from_pretrained('bert-base-cased')`.

The input of the BERT were the 2 features obtained with the already mentioned tokenizer ( `AutoTokenizer.from_pretrained('bert-base-cased')` ). 
   
Below, I reported the details about this model, the optimizer and the trained epochs.
    
&nbsp;    
    **<p align="center"> BERT with stopword and lemmatizer - Model </p>** 
   
   &nbsp;
    
| Layer (type)                   |   Output Shape    |  Param #  |    Connected to               |
|--------------------------------|---------------------|-------------|-----------------------------|    
| input_ids (InputLayer)         | [(None, 43)]      |     0     |                               |        
| attention_mask (InputLayer)    | [(None, 43)]      |     0     |                               |       
| tf_bert_model_1 (TFBertModel)  | TFBaseModelOutput | 108310272 |  input_ids[0][0]              |    
|                                |                   |           |  attention_mask[0][0]         |    
| global_max_pooling1d_1 (GlobalM| (None, 768)       |     0     |  tf_bert_model_1[1][0]        |    
| dense_3 (Dense)                | (None, 138)       |  106122   |  global_max_pooling1d_1[0][0] |    
| dropout_75 (Dropout)           | (None, 138)       |     0     |  dense_3[0][0]                |    
| dense_4 (Dense)                | (None, 28)        |   3892    |  dropout_75[0][0]             |    
| dense_5 (Dense)                | (None, 6)         |    174    |  dense_4[0][0]                |
          
&nbsp; 
   
**<p align="center"> BERT with stopword and lemmatizer - Optimer </p>** 
   
&nbsp;       
   
{'name': 'Adam',
 'clipnorm': 1.0,
 'learning_rate': 5e-05,
 'decay': 0.01,
 'beta_1': 0.9,
 'beta_2': 0.999,
 'epsilon': 1e-08,
 'amsgrad': False}       
    
    
&nbsp;    

**<p align="center"> BERT with stopword and lemmatizer - Training and validation accuracy  </p>** 
   
&nbsp; 
    
Epoch 1/3    
1334/1334 [==============================] - 959s 698ms/step - loss: 0.3806 - accuracy: 0.8639 - val_loss: 0.1853 - val_accuracy: 0.9315
    
Epoch 2/3
1334/1334 [==============================] - 924s 693ms/step - loss: 0.1362 - accuracy: 0.9416 - val_loss: 0.1656 - val_accuracy: 0.9305
    
Epoch 3/3
1334/1334 [==============================] - 933s 700ms/step - loss: 0.1113 - accuracy: 0.9482 - val_loss: 0.1550 - val_accuracy: 0.9375    
     
&nbsp; 
   
Thus, the model reached a **93.75 % validation accuracy** and 94.84 % on train dataset. **On test dataset, the model reached an accuracy of 92.95 %**, with a loss of 0.1699.

In this [notebook](https://github.com/Iron486/NLP_emotions_classifier/blob/main/Sentiment_prediction.ipynb) I applied the trained model in a more compact form to new sentences defined by the user. Running the script, it automatically applies the preprocessing steps and the evaluation, yielding the class prediction of the sentence as output. 
   
    
    
