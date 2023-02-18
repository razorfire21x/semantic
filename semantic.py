import spacy
#nlp = spacy.load('en_core_web_md')
nlp = spacy.load('en_core_web_sm')

word1 = nlp("cat")
word2 = nlp("monley")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana')

for token1 in tokens:
  for token2 in tokens:
    print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go", 
            "I've lost my car in my car", 
            "I'd like my boat back", 
            "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
  similarity = nlp(sentence).similarity(model_sentence)
  print(sentence + " - ", similarity)

#  --- Similarities between cat, monkey and banana: --- 
#  Cat and monkey seem to be compared as animals, apple and banana seemed to be compared as fruit.
#  Found it really strange that banana and monkey don't score higher on the similarity spectrum - 
#  I guess the data does not delve deeply into the culinary inclinations of primates. :P

#  --- 'en_core_we_sm' model ---
#  Apparantly cats and mokeys are very similar to apples. Banana's and monkeys still ranking low, 
#  Monkeys and cats seem to rank high together as animals.

