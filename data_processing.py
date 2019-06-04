import json
import random
import string

data = []
with open('Sarcasm_Headlines_Dataset.json','r') as file:
    for line in file:
        data.append(json.loads(line))

random.seed(642018)
random.shuffle(data)
training_set = data[:int(.8 * len(data))]
validate_set = data[int(.8 * len(data)):int(.9 * len(data))]
test_set = data[int(.9 * len(data)):]

def get_vocab_from_set(headlines, vocab_size=None):
    vocab = {}
    for headline in headlines:
        text = headline['headline'].replace('-',' ').split()
        for word in text:
            table = str.maketrans(dict.fromkeys(string.punctuation))
            word = word.translate(table)
            try:
                a = float(word)
                if '#NUMBER#' in vocab:
                    vocab['#NUMBER#'] += 1
                else:
                    vocab['#NUMBER#'] = 1
            except:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    return vocab

print(get_vocab_from_set(training_set))
#print(get_vocab_from_set(validate_set))
#print(get_vocab_from_set(test_set))
#print(get_vocab_from_set(data))
