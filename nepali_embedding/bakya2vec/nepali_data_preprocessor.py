import string
import os
import sys
import re

class Preprocessor:
    def __init__(self):
        self.this_dir, self.this_file = os.path.split(__file__)

    def clean_text(self,text):
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
        try:
            text = text.encode('ascii', errors='ignore').decode("utf-8")
            return text
        except:
            return text
    
    
    def sentence_tokenize(self, text):
        """This function tokenize the sentences
        
        Arguments:
            text {string} -- Sentences you want to tokenize
        
        Returns:
            sentence {list} -- tokenized sentence in list
        """
        sentences = text.strip().split(u"।")
        sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
        return sentences

    def word_tokenize(self, sentence, new_punctuation=[]):
        """This function tokenize with respect to word
        
        Arguments:
            sentence {string} -- sentence you want to tokenize
            new_punctuation {list} -- more punctutaion for tokenizing  default ['।',',',';','?','!','—','-']
        
        Returns:
            list -- tokenized words
        """
        punctuations = ['।', ',', ';', '?', '!', '—', '-', '.']
        if new_punctuation:
            punctuations = set(punctuations + new_punctuation)

        for punct in punctuations:
            sentence = ' '.join(sentence.split(punct))

        return sentence.split()

    def character_tokenize(self, word):
        """ Returns the tokenization in character level.
        
        Arguments:
            word {string} -- word to be tokenized in character level.
        
        Returns:
            [list] -- list of characters
        """
        try:
            import icu

        except:
            print("please install PyICU")
        
        temp_ = icu.BreakIterator.createCharacterInstance(icu.Locale())
        temp_.setText(word)
        char = []
        i = 0
        for j in temp_:
            s = word[i:j]
            char.append(s)
            i = j

        return char
