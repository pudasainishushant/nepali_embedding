from gensim.test.utils import common_texts
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nepali_tokenizer import Tokenizer
from scipy.spatial.distance import cosine 
from nltk.tokenize import word_tokenize

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
import re
import os
tok = Tokenizer()
import codecs
path = os.getcwd()
data_path = os.path.join(path,"data")
model_path = os.path.join(path,"model")


def clean_text(text):
    '''
    accepts the plain text and makes
    use of regex for cleaning the noise
    :param: text :type:str
    :return:cleaned text :type str
    '''
    text = text.lower()
    # text = ''.join([i for i in text if not i.isdigit()])
    text = re.sub(
        r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', '', text)
    text = re.sub(r'[|:}{=]', ' ', text)
    text = re.sub(r'[;]', ' ', text)
    text = re.sub(r'[\n]', ' ', text)
    text = re.sub(r'[\t]', ' ', text)
    text = re.sub(r'[[[]', ' ', text)
    text = re.sub(r'[]]]', ' ', text)
    text = re.sub(r'[-]', ' ', text)
    text = re.sub(r'[+]', ' ', text)
    text = re.sub(r'[*]', ' ', text)
    text = re.sub(r'[/]', ' ', text)
    text = re.sub(r'[//]', ' ', text)
    text = re.sub(r'[@]', ' ', text)
    text = re.sub(r'[,]', ' ', text)
    text = re.sub(r'[)]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('\n+', '\n', text)
    text = re.sub('\t+', '\t', text)
    text = [i.strip() for i in text.splitlines()]
    text = '\n'.join(text)
    text = re.sub('\n+', '\n', text)
    text = re.sub(r'[-]', ' ', text)
    text = re.sub(r'[(]', ' ', text)
    text = re.sub(' + ', ' ', text)
        # text = text.encode('ascii', errors='ignore').decode("utf-8")
    return text

#for alll file
# all_text = ""
# for root,subdir,files in os.walk(data_path):
#     for file in files:

#         textual_file_path = os.path.join(root,file)
#         textual_file_data = codecs.open(textual_file_path,"r", encoding='utf-8').read()
#         print("Length of individual text file",len(textual_file_data))
#         all_text += textual_file_data

# #for single file
single_textual_file_content = codecs.open("/home/info/Aakash/bert custom/dataset/ne.txt","r", encoding='utf-8').read()

print("Tokenizing")
sentences = tok.sentence_tokenize(single_textual_file_content)
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sentences)]

max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1,
                workers=24)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("bakya2vec.model")
print("Model Saved")

# from gensim.models.doc2vec import Doc2Vec

# model= Doc2Vec.load("d2v.model")
# #to find the vector of a document which is not in training data
# test_data = word_tokenize("I love chatbots".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# test_data_2 = word_tokenize("I love coding".lower())
# v2 = model.infer_vector(test_data_2)
# print("V1_infer", v2)

# cosine_similarity = 1 - cosine(v1,v2)
# print(cosine_similarity)
# to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('1')
# print(similar_doc)
# import pdb;pdb.set_trace()

# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.docvecs['1'])

