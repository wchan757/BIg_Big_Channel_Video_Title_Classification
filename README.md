## TVB big big channel's video title classification : File's structure
1. Code > full_model ------------------- the code for the model building with two embedding layer approach
2. Code > Word_Embedding > ------------- the code for different apporach of word embedding
3. big_big_cat --------------------------the csv file for the dataset of big big channel video title and its category 

## Overview
This is a NLP Text Classification task. I use the video's title of TVB Big Big Channel to predict the video's cateegory through Convolution Neural Network.
## Detail
### 1. Web Scraping
I used the Chrome extension "Web Scraper" to collect the data of video's title and its category in https://www.bigbigchannel.com.hk/tc/cat_bigbigtreasure. I have updaload the dataset with name of big_big_cat

### 2. Data Cleansing and preprocessing
I have filtered out some categorie because the number of the video in those categories are small. Before I do the word embedding, I used the library jieba to do the Chinese Segmentation and filter out any symptom that is not Chinese and English. Finally, I also filter out the stop word in Chinese(source:https://github.com/trec-kba/many-stop-words/tree/master/many_stop_words) and English(source: NLTK)

### 3. Word Embedding: Four Approachs
#### 3.1. Self-training embedding layer: 
My first attempt to do the word embedding is to train the embedding layer based ony my dataset. I do this because the video's title are mxied of Chinese and English and there are no pre-trained model for this type of data.  
#### 3.2. Pre-training embedding layer:
After I have discussed the problem with some AI enginners and my supervisor, I have tried different apporach for the word embedding. In this approach, I use the cantonese pre-trained model(source https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
and ignore the English because most of the words in the video's title are Cantonese. 
#### 3.3.Two Embedding Layer (1. Non-trainable Pre-training embedding layer 2. Trainable embediding layer for excluded word)
In this apprach, I use the custom word embedding layer.(source: https://stackoverflow.com/questions/46032700/concatenate-embedding-layers/46149153#46149153) This custom layer has two embedding layer and will let the included word(included in pre-trained vector)  go thourgh the pre-trained embedding layer(untrainable) and let the excluded word(not in pre-trained vector) go through another embedding layer(trainable) then use the function keras concatenate to combined those input and send to the Convolutional Layer. 
#### 3.4. One-hot encoding:
In this approach, I just encode each unique word into[1,0,0],[0,1,0].

### 4. Model Building:
I use Convolution Neural Network because this network have demonstrated good perofmrance in text classification(I have followed this tutorial throughout my task: https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/). In my first attempt, I only use 32 filter in the convolution layer and only one hidden layer without dropout. After I have discussed the problem with some AI enginners and my supervisor, I try to increase filter size from 32 to 128 in the convolution layer, add one more hidden layer and each hidden layer with 0.5 dropout rate.

### 5. Evaluation
I have used the same architecture to train the model with four word embedding apporachs. Each model has 10 epochs for training session. 

#### Model accuracy with different word embedding method in test data set (12 Categories) :
1. Self-Training:    55%
2. Pre-Training:     55%
3. Two Embedding:    58%
4. One-hot Encoding: 60%

It is suggested that self-training and pre-trainning haveing the lowest performance may be due to the problems of data loss and insufficient data size. In pre-training method, the word not in the pre-training vector will be excluded that may lose some important information in the sentence. Also, my dataset just has around 10k unique word and it may not be enough for the embedding layer to develop a complete picture of the relationship among the word. 

Surprisingly, the one-hot encoding outperform the two embedding layer approach. It may be due to the uniqueness of the dataset. One-hot encoding regards each word as unique symptom and ignore similarity among the word. If there are substantial amounts of the word that are exclusive to some categories and their similiar words seldomly appear in those categories, the one-hot encoding may outperform the word embedding method. 

Although one-hot encoding has the highest accuracy, its processing time are 10 times higher than other approach. As a result, taking accuracy and processing resource into consideration, the model with two embedding approach may be more appropriate for the real-word setting. 
