import numpy as np
import data_processing


class Perceptron():

    def __init__(self, feature_functions, data):
        self.feature_functions = feature_functions
        self.weights = self.sentence_to_vect('', self.feature_functions)
        self.data = data
        print(len(self.weights),self.weights)
        for key, value in self.weights.items():
            assert value == 0

    def sentence_to_vect(self, sentence, features):
        sentence_features = {}
        for feature in features:
            sentence_features.update(feature(sentence))
        return sentence_features

    def train(self, iterations=1):
        for i in range(iterations):
            for data_point in self.data.training_set:
                label = data_point['is_sarcastic']
                headline = data_point['headline']
                self.train_one_thing(self.sentence_to_vect(headline, self.feature_functions), label)

    def train_one_thing(self, sentence_dict, label):
        if label == 0:
            label = -1
        phi = self.classify(sentence_dict)
        loss = max(0, -label * phi)
        if phi < 0 and label == 1 or phi > 0 and label == 0:
            for key in self.weights:
                self.weights[key] += loss

    def classify(self, sentence_dict):
        sum = 0
        for key in sentence_dict:
            sum += sentence_dict[key] * self.weights[key]
        return sum

    def evaluate_effectiveness(self):
        true_positive = 0
        true_negative = 0
        false_negative = 0
        false_positive = 0
        total = 0
        for data_point in self.data.validate_set:
            label = data_point['is_sarcastic']
            eval = self.classify(self.sentence_to_vect(data_point['headline'], self.feature_functions))
            if label > 0 and eval > 0:
                true_positive += 1
            if label > 0 and eval < 0:
                false_negative += 1
            if label < 0 and eval > 0:
                false_positive += 1
            if label > 0 and eval > 0:
                true_negative += 1
            total += 1
        acc = (true_positive + true_negative) / total
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall /(precision + recall)
        print(acc, precision, recall, f1,sep = '\t')


if __name__ == "__main__":
    _data = data_processing.DataProcessing()
    FEATURES = [_data.num_dashes,
                _data.avg_word_length,
                _data.num_words,
                _data.num_possessive,
                _data.num_commas,
                _data.num_colons,
                _data.num_semicolons,
                _data.frequent_words
                ]
    perceptron = Perceptron(FEATURES, _data)
    perceptron.train(1)
    perceptron.evaluate_effectiveness()
    perceptron.train(1)
    perceptron.evaluate_effectiveness()
    perceptron.train(1)
    perceptron.evaluate_effectiveness()
    perceptron.train(1)

# class Perceptron(object):
#
#     def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
#         self.threshold = threshold
#         self.learning_rate = learning_rate
#         self.weights = np.zeros(no_of_inputs + 1)
#
#     def predict(self, inputs):
#         summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
#         if summation > 0:
#             activation = 1
#         else:
#             activation = 0
#         return activation
#
#     def train(self, training_inputs, labels):
#         for _ in range(self.threshold):
#             for inputs, label in zip(training_inputs, labels):
#                 prediction = self.predict(inputs)
#                 self.weights[1:] += self.learning_rate * (label - prediction) * inputs
#                 self.weights[0] += self.learning_rate * (label - prediction)
#
#
