import os

# root_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.expanduser('~/')


'''<------------Word2Vec model path------------------------------>'''
word2vec_model_sm_path = os.path.join(root_dir,'everest_nlp_data', 'sabda_to_vec_model')
word2vec_model_md_path = os.path.join(root_dir, 'everest_nlp_data', "sabda_to_vec_model_md")
# nepali_fasttext_model = os.path.join(root_dir,"sabda2vec/models/","cc.ne.300.vec")