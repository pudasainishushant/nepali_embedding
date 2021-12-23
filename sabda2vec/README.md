## Training custom word2vec model on nepai dataset.
This repo is for building a custom word2vec model for nepali language. Nepali data set has been collected from several sources like OSCAR corpus and scrapped from several nepali new portals. All of the data collected are in the data folder. 
## Steps to train the model
- Run pip install -r requirements.txt
- Run python train_sabda2vec.py

## Steps to use sabda2vec model
Currently one of the small model is trained using gensim word2vec. This model can be loaded using following lines of code
```python
from devanagari.sabda2vec.inference import Sabda2Vec
sabda2vec_obj = Sabda2Vec(model_name = "sabda2vec_sm")
#Get embedding of the token 
embedding = sabda2vec_obj.get_embedding("हार")
# Get top similar tokens 
top_similar = sabda2vec_obj.get_most_similar("हार",5)
# Get similarity between two tokens
similarity_score = sabda2vec_obj.get_similarity_between_tokens("हार","पराजय")
```