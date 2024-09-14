import nltk
import numpy as np
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem1(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem1(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag        

# words = ["organize","organizes","organizing"]
# stemmed_w=[stem1(w) for w in words]

# print(stemmed_w)

# a="Hi there guys!"
# print(a)
# a =tokenize(a)
# print(a)

# sentence = ["hello","how","are","you"]
# words = ["hi","hello","I","you","bye","thank","cool"]
# bag = bag_of_words(sentence,words)
# print(bag)