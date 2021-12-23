from simcse import SimCSE
model = SimCSE("Shushant/NepNewsBERT")
embeddings = model.encode("आकाश भाई ज्ञानी मन्छे हो")
print(embeddings)

sentences_a = ['एस्तो खत्तम चलचित्र्‍ मैले आहिले सम्म हेरेको थिन']
sentences_b = ['त्यो समान एक्दमै नराम्रो रहेछ']
similarities = model.similarity(sentences_a, sentences_b)
print(similarities)
