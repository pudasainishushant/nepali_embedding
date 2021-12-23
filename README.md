# nepali_embedding

Generate text embedding for nepali text

# Objective

Any NLP downstream tasks such as text classification, Named Entity Recognition performed on nepali textual data requires embedding of the words as the feature embedding matrix.
For english text, there are several ways we can generate embedding such as Word2Vec, Glove, BERT and so on from several open source NLP libraries. However, there seems to be no open source library for such tasks in nepali text. We aim to provide accurate word embedding for nepali text through several NLP based architectures so that it can be used in further NLP downstream tasks in nepali language.

# Models used overview

We have developed state of the art models for embedding generation for nepali text.

- Sabda2Vec
- Bakya2Vec
- NepBERT

# Usage

## Word_embedding

Use this script to get embedding of a word (sabda).

```python
from nepali_embedding.sabda2vec.inference import Sabda2Vec
sabda2vec_obj = Sabda2Vec(model_name = "sabda2vec_sm")
#Get embedding of the token
embedding = sabda2vec_obj.get_embedding("हार")
# Get top similar tokens
top_similar = sabda2vec_obj.get_most_similar("हार",5)
# Get similarity between two tokens
similarity_score = sabda2vec_obj.get_similarity_between_tokens("हार","पराजय")
```

```
    sabda_to_vec_model: https://www.dropbox.com/s/xkd29spkozoavhk/sabda_to_vec_model?dl=0
    sabda_to_vec_model_md: https://www.dropbox.com/s/55m5q4h5ys1l4np/sabda_to_vec_model_md?dl=0
```