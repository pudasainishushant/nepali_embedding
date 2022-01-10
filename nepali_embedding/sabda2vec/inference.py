import sys
sys.path.append("..")

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

from nepali_embedding.config import word2vec_model_sm_path,word2vec_model_md_path
class Sabda2Vec():
    def __init__(self,model_name):
        """
        Constructor for loading sabda2vec model and getting vocab list
        """
        if model_name == "sabda2vec_sm":
            self.model_name = KeyedVectors.load(word2vec_model_sm_path)
            print("INFO :: Small model loaded")
        elif model_name == "sabda2vec_md":
            self.model_name = KeyedVectors.load(word2vec_model_md_path)
            print("INFO :: Medium model loaded")
        # elif model_name == "sabda2vec_lg":
        #     try:
        #         self.model_name = KeyedVectors.load(word2vec_model_lg_path)
        #         print("Large model loaded")
        #     except OSError as e:
        #         self.model_name = KeyedVectors.load(word2vec_model_md_path)
        #         print("Medium model loaded")
        # elif model_name == "fasttext_model":
        #     try:
        #         self.model_name = KeyedVectors.load(fasttext_model_path)
        #         print("Faststext model loaded")
        #     except OSError as e:
        #         self.model_name = KeyedVectors.load(word2vec_model_md_path)
        #         print("Medium model loaded")
        # elif model_name == "nepali_nlp_gensim_model":
        #     try:
        #         self.model_name == KeyedVectors.load(nepali_nlp_gensim_model_path)
        #         print("Nepali nlp model loaded")
        #     except OSError as e:
        #         self.model_name = KeyedVectors.load(word2vec_model_md_path)
        #         print("Medium model loaded")
        else:
            self.model_name = KeyedVectors.load(word2vec_model_sm_path)
            print("INFO :: Small model loaded")

    
    def get_most_similar(self,token,num_of_similar_tokens):
        """
        Get most similar tokens from the loaded sabda2vec model.
        params: token: str
                num_of_similar_tokens: int
        return: similar_tokens : list of tuples
        """
        most_similar = self.model_name.wv.most_similar_cosmul(positive=[token], topn=num_of_similar_tokens)
        return most_similar

    def get_embedding(self,token):
        """
        Get the vector embedding of the token from the loaded model.
        params: token: str
        return: embedding : numpy_array
        """
        try:
            token_embedding = self.model_name.wv[token]
        except KeyError:
            print("The word  does not appear in this model")
        return token_embedding

    def get_similarity_between_tokens(self,token1,token2):
        """
        Get the similarity between two nepali tokens
        params: token1 : str
                token2: str
        return: simalarity_score: float
        """
        try:
            similarity_score = self.model_name.wv.similarity(token1,token2)
            return similarity_score
        except KeyError:
            print("One of the word  does not appear in this model")
            return 0

if __name__ == "__main__":
    s2v_object = Sabda2Vec("sabda2vec_md")
    test_word_1 = "सुसान्त"
    test_word_2 = "भालु"
    test_word_3 = "जित"
    most_simialar = s2v_object.get_most_similar(test_word_2,5)
    print("Top 10 words most similar to {} are \n {}".format(test_word_3,most_simialar))
    sim = s2v_object.get_similarity_between_tokens(test_word_1,test_word_2)
    sim2 = s2v_object.get_similarity_between_tokens(test_word_2,test_word_3)
    print("sim",sim)
    print("sim2",sim2)
