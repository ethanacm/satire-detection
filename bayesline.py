from collections import defaultdict
import operator
import data_processing
import string
import math

class BayesClassifier:
    
    def __init__(self, history = 1, k = 1.0, vocab_size = 3000):
        self.unigram_dict = {}
        self.vocab = set(['<UNK>'])
        self.data = data_processing.DataProcessing()
        self.get_vocab(3000)
        self.history = history
        self.k = k
        self.process_training_set_unigrams()

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
            self.vocab.add(sorted_vocab[i][0])
        
    def process_training_set_unigrams(self):
        self.unigram_dict[0] = defaultdict(lambda:0)
        self.unigram_dict[1] = defaultdict(lambda:0)
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
        if log_probs[0] >= log_probs[1]:
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

def get_accuracy(true, predicted):
    right = 0.0
    for i in range(0, len(predicted)):
        if true[i] == predicted[i]:
            right += 1
    return right/len(predicted)

def get_precision(true, predicted):
    tp = 0.0
    fp = 0.0
    for i in range(0, len(predicted)):
        if true[i] == 1 and predicted[i] == 1:
            tp += 1
        elif true[i] == 1 and predicted[i] == 0:
            fp += 1
    if tp == 0 and fp == 0:
        return 0
    return tp/(tp + fp)

def get_recall(true, predicted):
    tp = 0.0
    fn = 0.0
    for i in range(0, len(predicted)):
        if true[i] == 1 and predicted[i] == 1:      
            tp += 1
        elif true[i] == 0 and predicted[i] == 1:
            fn += 1
    if tp == 0 and fn == 0:
        return 0
    return tp/(tp + fn)

def get_f1(true, predicted):
    precision = get_precision(true, predicted)
    recall = get_recall(true, predicted)
    if precision == 0 and recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def eval_metrics(labels, preds):
    kacc = round(get_accuracy(labels, preds), 4)
    kf1 = round(get_f1(labels, preds), 4)
    kpre = round(get_precision(labels, preds), 4)
    krec = round(get_recall(labels, preds), 4)
    print ('Accuracy:', kacc, 'Precision:', kpre, 'Recall:', krec, 'F1:', kf1, sep = '\t')


if __name__ == "__main__":
    a= BayesClassifier()
    b,c = a.bayes_classify(a.data.validate_set)
    e,f = a.bayes_classify(a.data.test_set)
    print ("Bayes on Validation Set")
    eval_metrics(b, c)
    print("\nBayes on Test Set")
    eval_metrics(e,f)