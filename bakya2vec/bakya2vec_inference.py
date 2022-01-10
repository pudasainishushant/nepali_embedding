from gensim.models.doc2vec import Doc2Vec
from nepali_data_preprocesser import Preprocessor
from scipy.spatial.distance import cosine 

preprocessor_obj = Preprocessor()


model= Doc2Vec.load("d2v_small.model")
#to find the vector of a document which is not in training data
test_data = preprocessor_obj.word_tokenize("यो समान धेरै नराम्रो रैछ मलाई त लस्त नराम्रो लाग्यो ।".lower())
v1 = model.infer_vector(test_data)
test_data_2 = preprocessor_obj.word_tokenize("क्या राम्रो फिल्म रैछ म त दोराएर हेर्न आउने हो ।".lower())
v2 = model.infer_vector(test_data_2)

sims = model.dv.most_similar([v1], topn=len(model.dv))
print("V1_infer", v1)

similarity_score = cosine(v1, v2)
import pdb;pdb.set_trace()

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])