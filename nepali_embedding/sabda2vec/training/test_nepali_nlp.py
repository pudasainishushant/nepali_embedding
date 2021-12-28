from nepali_tokenizer import Tokenizer
tok = Tokenizer()
text = "रामले आज पुरस्कार ग्रहण गर्यो । पहिले बर्सको भन्दा यो वर्ष चाडैनै ग्रहण लाग्यो । खुल्ला आखाले ग्रहण हेरेमा आँखा मा धेरै असर पर्ने बिज्ञा हरुले बताएका छन् ।"
sentences = tok.sentence_tokenize(text) #To tokenize sentence
tokens = tok.word_tokenize(text)
print(sentences)
print(tokens)

