# nepali_embedding
Generate text embedding for nepali text

# Objective
Any NLP downstream tasks such as text classification, Named Entity Recognition performed on nepali textual data requires embedding of the words as the feature embedding matrix.
For english text, there are several ways we can generate embedding such as Word2Vec, Glove, BERT and so on from several open source NLP libraries. However, there seems to be no open source library for such tasks in nepali text. We aim to provide accurate word embedding for nepali text through several NLP based architectures so that it can be used in further NLP downstream  tasks in nepali language.


# Models used overview
We have developed state of the art models for embedding generation for nepali text. 
- Sabda2Vec
- Bakya2Vec
- NepBERT

# Usage

## Word_embedding

Use this script to get embedding of a word (sabda).

## To generate embedidng of a word/token from custom trained word2vec nepali model
```python
from nepali_embedding.sabda2vec.inference import Sabda2Vec
sabda2vec_obj = Sabda2Vec(model_name = "sabda2vec_sm")
#Get embedding of the token 
embedding = sabda2vec_obj.get_embedding("हार")
```

## To generate embedidng of a word/token from custom trained doc2vec nepali model
```python
from nepali_embedding.bakya2vec.inference import Bakya2Vec
bakya2vec_obj = Bakya2Vec(model_name = "bakya2vec_sm")
#Get embedding of the token 
embedding = bakya2vec_obj.get_embedding("हार")
```

## To generate embedidng of a word/token from NepBERT model
```python
from nepali_embedding.nepBERT.inference import NepBERT
nepbert = NepBERT()
#For sentence embedding
embedding1 = nepbert.get_word_embedding_bert("हार")
```

