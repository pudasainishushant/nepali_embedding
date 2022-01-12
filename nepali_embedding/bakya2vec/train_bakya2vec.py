import codecs
import time
import pandas as pd
from nepali_data_preprocesser import Preprocessor
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

preprocessor_obj = Preprocessor()

single_textual_file_content = codecs.open("/home/info/Aakash/bert custom/dataset/ne.txt","r", encoding='utf-8').read()
print("Cleaning text")
cleaned_text = preprocessor_obj.clean_text(single_textual_file_content)
print("Converting to sentences")
sentences = preprocessor_obj.sentence_tokenize(single_textual_file_content)
print("Converting to words")
words = [preprocessor_obj.word_tokenize(sent) for sent in sentences]
final_words = [x for x in words if x]
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(final_words)]
print("Total number of sentences in training data",len(tagged_data))
max_epochs = 10
vec_size = 20
alpha = 0.025

start = time.time()
model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1,workers=24,compute_loss=True)
model.build_vocab(tagged_data)
print("Training started")

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

end = time.time()
print("Total training time",end-start)
model.save("d2v_small.model")
print("Model Saved")

# from gensim.models.doc2vec import Doc2Vec

# model= Doc2Vec.load("d2v.model")
# #to find the vector of a document which is not in training data
# test_data = word_tokenize("I love chatbots".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# # to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('1')
# print(similar_doc)


# # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.docvecs['1'])