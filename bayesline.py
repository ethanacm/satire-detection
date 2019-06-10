from collections import defaultdict
import operator
import data_processing
import string
import math

class BayesClassifier:
    
    def __init__(self, history = 1, k = 1.0, vocab_size = 3000):
        self.unigram_dict = { 1:{'<UNK>':0}, 0:{'<UNK>':0} }
        self.ngram_dict = {1:{},0:{}}
        self.vocab = set(['<s>', '</s>', '<UNK>'])
        self.data = data_processing.DataProcessing()
        self.get_vocab(3000)
        self.history = history
        self.k = k
        self.process_training_set_unigrams()
        self.process_training_set_ngrams(history)

    def get_vocab(self, vocab_size):
        vocab_dict = defaultdict(lambda: 0)
        for line in self.data.training_set:
            headline = line['headline']
            for word in headline.replace('-', ' ').split():
                table = str.maketrans(dict.fromkeys(string.punctuation))
                word = word.translate(table)
                vocab_dict[word] += 1
        sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1), reverse = True)
        for i in range(0, vocab_size - len(self.vocab)):
            self.vocab.add(sorted_vocab[0])
        
    def process_training_set_unigrams(self):
        for line in self.data.training_set:
            label = line['is_sarcastic']
            headline = line['headline']
            for word in headline.replace('-', ' ').split():
                table = str.maketrans(dict.fromkeys(string.punctuation))
                word = word.translate(table)
                if word in self.vocab:
                    self.unigram_dict[label][word] += 1
                else:
                    self.unigram_dict[label]['<UNK>'] += 1

    def compute_probability_unigram(self, word):
        probabilities = []
        for i in range(0,2):
            word_count_smoothed = self.unigram_dict[i][word] + self.k 
            smoothed_denom = sum(self.unigram_dict[i].values()) + self.k * len(self.vocab)
            probabilities.append(word_count_smoothed / smoothed_denom)
        return probabilities

    def classify_unigrams(self, sentence):
        log_probs = [0, 0]
        words = sentence.replace('-', ' ').split()
        tokens = []
        for word in words:
            table = str.maketrans(dict.fromkeys(string.punctuation))
            word = word.translate(table)
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append('<UNK>')
        for tok in tokens:
            probs = self.compute_probability_unigram(tok)
            log_probs[0] += math.log(probs[0])
            log_probs[1] += math.log(probs[1])
        if log_probs[0] > log_probs[1]:
            return 0
        else:
            return 1

    def bayes_classify(self, dataset):
        preds = []
        labs = []
        for headline in dataset:
            preds.append(self.classify_unigrams(headline['headline']))
            labs.append(headline['is_sarcastic'])
        return labs, preds


if __name__ == "__main__":
    BayesClassifier()