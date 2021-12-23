import os
import re
import string
import codecs
from nepali_tokenizer import Tokenizer
from gensim.models import Word2Vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


tok = Tokenizer()
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
single_textual_file_content = codecs.open("/media/info/New Volume/sabda2vec/data/ne.txt","r", encoding='utf-8').read()

# print(single_textul_file)

# print("Cleaning text")
# cleaned_text = clean_text(single_textual_file_content)
print("Tokenizing")
sentences = tok.sentence_tokenize(single_textual_file_content)
token_list = [tok.word_tokenize(sent.translate(str.maketrans(
    '', '', string.punctuation))) for sent in sentences]

print("Training")
model = Word2Vec(token_list, min_count=2, window=5,
                 sample=6e-5, alpha=0.03, min_alpha=0.007, workers=24,compute_loss=True)

model.train(token_list, total_examples=model.corpus_count,
            epochs=50, report_delay=1)
training_loss = model.get_latest_training_loss()
print(training_loss)
# words = list(model.wv.vocab)
# access vector for one word
# save model
model.save(model_path+'/sabda_to_vec_model_ne')

