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


    def process_training_set_ngrams(self, history):
        for line in self.data.training_set:
            label = line['is_sarcastic']
            headline = line['headline'] 
            words = headline.replace('-', ' ').split()
            processed_words = ['<s>'] * history
            for word in words:
                table = str.maketrans(dict.fromkeys(string.punctuation))
                word = word.translate(table)
                if word in self.vocab:
                    processed_words.append(word)
                else:
                    processed_words.append('<UNK>')
            processed_words.append('</s>')
            for i in range(len(processed_words) - history):
                key = tuple( processed_words[i:i+history] )
                if key in self.ngram_dict[label]:
                    if processed_words[history + i] in self.ngram_dict[label][key]:
                        self.ngram_dict[label][key][processed_words[history + i]] += 1
                    else:
                        self.ngram_dict[label][key][processed_words[history + i]] = 1
                else:
                    self.ngram_dict[label][key] = defaultdict(lambda: 0)
                    self.ngram_dict[label][key][processed_words[history + i]] = 1

    def compute_probability_ngrams(self, context, word):
        probs = []
        for i in range(0,2):
            context_dict = self.ngram_dict[i][context]
            word_count_smoothed = context_dict[word] + self.k 
            smoothed_denom = sum(context_dict.values()) + self.k * len(self.vocab)
            probs.append(word_count_smoothed / smoothed_denom)
        return probs

    def classify_ngrams(self, sentence):
        log_probs = [0, 0]
        tokens = ['<s>'] * (self.history)
        words = sentence.replace('-', ' ').split()
        for word in words:
            table = str.maketrans(dict.fromkeys(string.punctuation))
            word = word.translate(table)
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append('<UNK>')
        tokens.append('</s>')
        for i in range(self.history, len(tokens)):
            probs = self.compute_probability_ngrams(tuple(tokens[i - self.history:i]), tokens[i])
            log_probs[0] += math.log(probs[0])
            log_probs[1] += math.log(probs[1])
        if log_probs[0] > log_probs[1]:
            return 0
        else:
            return 1

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

    def validation_data(self):
        acc = 0
        for headline in self.data.validate_set:
            label = headline['is_sarcastic']
            if self.classify_unigrams(headline['headline']) == label:
                acc += 1
        print (acc/len(self.data.validate_set))

    def validation_data2(self):
        acc = 0
        for headline in self.data.validate_set:
            label = headline['is_sarcastic']
            if self.classify_ngrams(headline['headline']) == label:
                acc += 1
        print (acc/len(self.data.validate_set))




if __name__ == "__main__":
    a = BayesClassifier()
    a.validation_data()
    a.validation_data2()