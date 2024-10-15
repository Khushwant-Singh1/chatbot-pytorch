import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Download the necessary NLTK data
nltk.download('punkt')

stemmer = PorterStemmer()

def tokenzie(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # Stem the words in the tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, w in enumerate(all_words):
        if w in sentence_words:  # Check against stemmed words
            bag[idx] = 1

    return bag
