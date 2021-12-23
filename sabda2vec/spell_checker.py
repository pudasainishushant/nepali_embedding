import os
import sys
sys.path.append("..")
import jellyfish
# from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
# from config import word2vec_model_sm_path,word2vec_model_md_path,word2vec_model_lg_path,fasttext_model_path, nepali_nlp_gensim_model_path
from everest_nlp.config import word2vec_model_md_path,word2vec_model_sm_path,word2vec_model_lg_path,fasttext_model_path, nepali_nlp_gensim_model_path
import re

print("Model path",word2vec_model_md_path)

class SpellChecker():
    def __init__(self,model_name):
        # self.model_name = KeyedVectors.load(os.getcwd()+'/word2vec/models/sabda_to_vec_model_md')
        """
        Constructor for loading sabda2vec model and getting vocab list
        """
        if model_name == "sabda2vec_sm":
            self.model_name = KeyedVectors.load(word2vec_model_sm_path)
            print("Small model loaded")
        elif model_name == "sabda2vec_md":
            self.model_name = KeyedVectors.load(word2vec_model_md_path)
            print("Medium model loaded")
        elif model_name == "sabda2vec_lg":
            try:
                self.model_name = KeyedVectors.load(word2vec_model_lg_path)
                print("Large model loaded")
            except OSError as e:
                self.model_name = KeyedVectors.load(word2vec_model_md_path)
                print("Medium model loaded")
        elif model_name == "fasttext_model":
            try:
                self.model_name = KeyedVectors.load(fasttext_model_path)
                print("Faststext model loaded")
            except OSError as e:
                self.model_name = KeyedVectors.load(word2vec_model_md_path)
                print("Medium model loaded")
        elif model_name == "nepali_nlp_gensim_model":
            try:
                self.model_name == KeyedVectors.load(nepali_nlp_gensim_model_path)
                print("Nepali nlp model loaded")
            except OSError as e:
                self.model_name = KeyedVectors.load(word2vec_model_md_path)
                print("Medium model loaded")
        else:
            self.model_name = KeyedVectors.load(word2vec_model_sm_path)
            print("Small model loaded")
        self.vocab = list(self.model_name.wv.vocab.keys())
        self.vocab = [self.clean_text(v) for v in self.vocab]

    def clean_text(self,text):
        text = re.sub(r"[०१२३४५६७८९‘“’”]",'',text)
        return text

    def correct_words(self,text):
        text = self.clean_text(text)
        jaro_sim = [jellyfish.jaro_distance(i,text) for i in self.vocab]
        corrected_words = []
        for i in range(10):
            max_distance = max(jaro_sim)
            index_of_similar_text = jaro_sim.index(max_distance)
            corrected_words.append(self.vocab[index_of_similar_text])
            jaro_sim.pop(index_of_similar_text)
            self.vocab.pop(index_of_similar_text)
        return corrected_words

if __name__ == '__main__':
    spc = SpellChecker("sabda2vec_md")
    text = 'रमण'
    corrected_words = spc.correct_words(text)
    for correct in corrected_words:
        print(correct)