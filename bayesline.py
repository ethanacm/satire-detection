from collections import defaultdict
import data_processing

class BayesClassifier:
    
    def __init__(ngram = 1, k = 0.0):
        self.unigram_dict = { 1:defaultdict(lambda:k), 0:{defaultdict(lambda:k)} }
        self.ngram_dict = {1:{},0:{}}
        self.data = data_processing.DataProcessing()
        self.ngram = ngram
        self.k = k
        self.process_training_set_unigrams(self.data.training_set)
        self.process_training_set_ngrams(self.ngram)

    def process_training_set_unigrams(self):
        for line in self.data.training_set:
            label = line['is_sarcastic']
            headline = line['headline']
            for word in headline.replace('-', ' ').split():
                table = str.maketrans(dict.fromkeys(string.punctuation))
                word = word.translate(table)
                self.unigram_dict[label][word] += 1

    def compute_probability_unigram(self, word):
        probabilities = []
        for i in range(0,2):
            word_count_smoothed = self.unigram_dict[i][word] + self.k 
            smoothed_denom = sum(self.unigram_dict[i].values()) + self.k * len(self.unigram_dict[i])
            probabilities.append(word_count_smoothed / smoothed_denom)
        return probabilities


    def process_training_set_ngrams_(self, ngram):
        for line in self.data.training_set:
            label = line['is_sarcastic']
            headline = line['headline'] 
            words = headline.replace('-', ' ').split()
            new_words = ['<s>'] * (ngram - 1)
            for word in words:
                table = str.maketrans(dict.fromkeys(string.punctuation))
                word = word.translate(table)
                new_words.append(word)
            new_words.append('</s>')

        return None

    def compute_probability_ngrams(self, word):
        return None
                





if __name__ == "__main__":
    

