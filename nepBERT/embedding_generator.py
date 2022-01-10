import torch
from scipy.spatial.distance import cosine 


from transformers import AutoTokenizer, AutoModelForMaskedLM

class NepBERT():
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")
        self.model = AutoModelForMaskedLM.from_pretrained("Shushant/nepaliBERT",output_hidden_states=True)
    
    def get_word_embedding_bert(self,input_text):
        marked_text = " [CLS] " + input_text + " [SEP] "
        tokenized_text = self.tokenizer.tokenize(marked_text)
        for i, token_str in enumerate(tokenized_text):
            print (i, token_str)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens) 
        
        # Convert inputs to Pytorch tensors
        tokens_tensors = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        with torch.no_grad():
            outputs = self.model(tokens_tensors, segments_tensors)
            # removing the first hidden state
            # the first state is the input state 
            hidden_states = outputs.hidden_states
        
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)


        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)
        return token_vecs_sum

    def get_bert_embedding_sentence(self,input_sentence):

        marked_text = " [CLS] " + input_sentence + " [SEP] "
        tokenized_text = self.tokenizer.tokenize(marked_text)
        for i, token_str in enumerate(tokenized_text):
            print (i, token_str)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens) 
        
        # Convert inputs to Pytorch tensors
        tokens_tensors = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        with torch.no_grad():
            outputs = self.model(tokens_tensors, segments_tensors)
            # removing the first hidden state
            # the first state is the input state 
            # import pdb;pdb.set_trace()
            hidden_states = outputs.hidden_states
            # second_hidden_states = outputs[2]
        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # import pdb;pdb.set_trace()
        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding


if __name__== "__main__":
    import time
    start = time.time()
    test_sentence = 'आकाश भाई ज्ञानी मन्छे हो'
    test_sentence_2 = "त्यो समान एक्दमै नराम्रो रहेछ"
    test_sentence_3 = "एस्तो खत्तम चलचित्र्‍ मैले आहिले सम्म हेरेको थिन"
    # embedding1 = get_word_embedding_bert(test_sentence,tokenizer,x)
    # embedding2 = get_word_embedding_bert(test_sentence_2,tokenizer,x)
    # embedding3 = get_word_embedding_bert(test_sentence_3,tokenizer,x)
    
    # #For word embedding
    # embedding1 = get_word_embedding_bert(test_sentence,tokenizer,x)
    # embedding2 = get_word_embedding_bert(test_sentence_2,tokenizer,x)
    # embedding3 = get_word_embedding_bert(test_sentence_3,tokenizer,x)
    # embedding_1_haar = embedding1[6]
    # embedding_2_haar = embedding2[13]
    # embedding_3_haar = embedding3[7]
    # cos_dist_different = 1 - cosine(embedding_1_haar,embedding_2_haar)
    # cos_dist_same = 1 - cosine(embedding_2_haar,embedding_3_haar)
    # print("Difference",cos_dist_different)
    # print("Similar",cos_dist_same)
    # end = time.time()
    # print("Time",end-start)
    nepbert = NepBERT()
    #For sentence embedding
    embedding1 = nepbert.get_bert_embedding_sentence(test_sentence,)
    embedding2 = nepbert.get_bert_embedding_sentence(test_sentence_2)
    embedding3 = nepbert.get_bert_embedding_sentence(test_sentence_3)
    # import pdb;pdb.set_trace()
    cos_dist_different = 1 - cosine(embedding1,embedding2)
    cos_dist_same = 1 - cosine(embedding2,embedding3)
    print("Difference",cos_dist_different)
    print("Similar",cos_dist_same)
    end = time.time()
    print("Time",end-start)




# from transformers import pipeline

# feature_extraction = pipeline('feature-extraction', model=mlm_model, tokenizer=tokenizer)
# features = feature_extraction(["आकाश भाई ज्ञानी मन्छे हो",
#                                "त्यो समान एक्दमै नराम्रो रहेछ",
#                                "एस्तो खत्तम चलचित्र्‍ मैले आहिले सम्म हेरेको थिन"])


# # import pdb;pdb.set_trace()
# cos_dist_same = 1 - cosine(torch.mean(torch.Tensor(features[0][0]), dim=0),torch.mean(torch.Tensor(features[1][0]), dim=0))
# cos_dist_different = 1 - cosine(torch.mean(torch.Tensor(features[1][0]), dim=0),torch.mean(torch.Tensor(features[2][0]), dim=0))
# print("Difference",cos_dist_different)
# print("Similar",cos_dist_same)


# from transformers import pipeline

# fill_mask = pipeline(
#     "fill-mask",
#     model=x,
#     tokenizer=tokenizer
# )


# MASK_TOKEN = tokenizer.mask_token

# print(fill_mask("कांग्रेसले चुनाबमा एमालेसंग भारि मातले {} भेहोर्यो".format(MASK_TOKEN)))