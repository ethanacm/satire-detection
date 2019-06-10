import numpy as np
import data_processing
import string
import sklearn
from sklearn import tree
from sklearn import neighbors
import bayesline

class Perceptron():

    def __init__(self, feature_functions, data):
        self.feature_functions = feature_functions
        self.weights = self.sentence_to_vect('', self.feature_functions)
        self.data = data
        self.bias = 0
        #print(len(self.weights),self.weights)
        for key, value in self.weights.items():
            assert value == 0

    def sentence_to_vect(self, sentence, features):
        sentence_features = {}
        for feature in features:
            sentence_features.update(feature(sentence))
        return sentence_features

    def sentence_to_vector(self, sentence, features):
        sentence_features = {}
        for feature in features:
            sentence_features.update(feature(sentence))
        sorted_keys = sorted(sentence_features)
        arr = np.ndarray(len(sentence_features))
        for i, key in enumerate(sorted_keys):
            arr[i] = sentence_features[key]
        return arr

    def fit_knn(self, neighs = 7):
        arr = None
        labs = []
        print('assembling data for fitting knn')
        for index, headline in enumerate(self.data.training_set):
            title = headline['headline']
            label = headline['is_sarcastic']
            labs.append(label)
            vect = self.sentence_to_vector(title, self.feature_functions)
            if arr is None:
                size = (len(self.data.training_set), len(vect) )
                arr = np.ndarray(size)
            arr[index] = vect
        print('fitting knn')
        knn = neighbors.KNeighborsClassifier(n_neighbors = neighs)
        knn.fit(arr, labs)
        self.knn = knn

    def classify_knn(self):
        arr = None
        valid_labels = []
        test_labels = []
        arr2 = None
        for index, headline in enumerate(self.data.validate_set):
            title = headline['headline']
            label = headline['is_sarcastic']
            valid_labels.append(label)
            vect = self.sentence_to_vector(title, self.feature_functions)
            if arr is None:
                size = (len(self.data.validate_set), len(vect) )
                arr = np.ndarray(size)
            arr[index] = vect
        for index, headline in enumerate(self.data.test_set):
            title = headline['headline']
            label = headline['is_sarcastic']
            test_labels.append(label)
            vect = self.sentence_to_vector(title, self.feature_functions)
            if arr2 is None:
                size = (len(self.data.test_set), len(vect) )
                arr2 = np.ndarray(size)
            arr2[index] = vect
        valid_preds = self.knn.predict(arr)
        test_preds = self.knn.predict(arr2)
        return valid_preds, test_preds, valid_labels, test_labels

    def majority_vote_prediction(self,preds1, preds2, preds3):
        preds = []
        for i in range(0,len(preds1)):
            count0 = 0
            count1 = 0
            if preds1[i] == 0:
                count0 += 1
            else:
                count1 += 1
            if preds2[i] == 0:
                count0 += 1
            else:
                count1 += 1
            if preds3[i] == 0:
                count0 += 1
            else:
                count1 += 1
            if count0 > count1:
                preds.append(0)
            else:
                preds.append(1)
        return preds

    def fit_decision_tree(self):
        arr = None
        labs = []
        print('assembling data for fitting tree')
        for index, headline in enumerate(self.data.training_set):
            title = headline['headline']
            label = headline['is_sarcastic']
            labs.append(label)
            vect = self.sentence_to_vector(title, self.feature_functions)
            if arr is None:
                size = (len(self.data.training_set), len(vect) )
                arr = np.ndarray(size)
            arr[index] = vect
        print('fitting tree')
        tree = sklearn.tree.DecisionTreeClassifier(criterion='entropy')
        tree.fit(arr, labs)
        self.tree = tree

    def classify_decision_tree(self):
        arr = None
        valid_labels = []
        test_labels = []
        arr2 = None
        for index, headline in enumerate(self.data.validate_set):
            title = headline['headline']
            label = headline['is_sarcastic']
            valid_labels.append(label)
            vect = self.sentence_to_vector(title, self.feature_functions)
            if arr is None:
                size = (len(self.data.validate_set), len(vect) )
                arr = np.ndarray(size)
            arr[index] = vect
        for index, headline in enumerate(self.data.test_set):
            title = headline['headline']
            label = headline['is_sarcastic']
            test_labels.append(label)
            vect = self.sentence_to_vector(title, self.feature_functions)
            if arr2 is None:
                size = (len(self.data.test_set), len(vect) )
                arr2 = np.ndarray(size)
            arr2[index] = vect
        valid_preds = self.tree.predict(arr)
        test_preds = self.tree.predict(arr2)
        return valid_preds, test_preds, valid_labels, test_labels
        
    def train(self, iterations=1):
        for i in range(iterations):
            for data_point in self.data.training_set:
                label = data_point['is_sarcastic']
                headline = data_point['headline']
                self.train_one_thing(self.sentence_to_vect(headline, self.feature_functions), label)

    def train_one_thing(self, sentence_dict, label):
        if label == 0:
            label = -1
        activation = self.classify(sentence_dict)
        # loss = max(0, -label * activation)
        if activation * label <= 0:
            for key in self.weights:
                # self.weights[key] += loss
                self.weights[key] += label * sentence_dict[key]
            self.bias += label

    def classify(self, sentence_dict):
        sum = 0
        for key in sentence_dict:
            sum += sentence_dict[key] * self.weights[key] + self.bias
        return sum

    def predict(self, data_set):
        preds = []
        for data_point in data_set:
            label = data_point['is_sarcastic']
            if label == 0:
                label = -1
            eval = self.classify(self.sentence_to_vect(data_point['headline'], self.feature_functions))
            if eval > 0:
                preds.append(1)
            else:
                preds.append(0)
        return preds


    def evaluate_effectiveness(self, data_set):
        true_positive = 0
        true_negative = 0
        false_negative = 0
        false_positive = 0
        strange = 0
        total = 0
        preds = []
        for data_point in data_set:
            label = data_point['is_sarcastic']
            if label == 0:
                label = -1
            eval = self.classify(self.sentence_to_vect(data_point['headline'], self.feature_functions))
            if eval > 0:
                preds.append(1)
            else:
                preds.append(0)
            # if eval < 0:
            #     print(data_point['headline'])
            if label > 0 and eval > 0:
                true_positive += 1
            if label > 0 and eval < 0:
                false_negative += 1
            if label < 0 and eval > 0:
                false_positive += 1
            if label < 0 and eval < 0:
                true_negative += 1
            if eval == 0:
                strange += 1
            total += 1
        assert true_positive + true_negative + false_negative + false_positive + strange == total
        acc = (true_positive + true_negative) / total
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall /(precision + recall)
        print('acc', acc, 'precision', precision, 'recall', recall, 'f1', f1,sep = '\t')
        return preds

def evaluate_baseline(dataset, word_counts):
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    strange = 0
    total = 0

    for headline in dataset:
        sarcastic_occurances = 0
        serious_occurrances = 0
        text = headline['headline'].replace('-', ' ').split()
        for word in text:
            table = str.maketrans(dict.fromkeys(string.punctuation))
            word = word.translate(table)
            sarcastic_occurances += word_counts[1].get(word, 0)
            serious_occurrances += word_counts[0].get(word, 0)
        eval = 1 if sarcastic_occurances > serious_occurrances else 0
        label = headline['is_sarcastic']
        if label > 0 and eval > 0:
            true_positive += 1
        if label > 0 and eval <= 0:
            false_negative += 1
        if label <= 0 and eval > 0:
            false_positive += 1
        if label <= 0 and eval <= 0:
            true_negative += 1
        # if eval == 0:
        #     strange += 1
        total += 1
    assert true_positive + true_negative + false_negative + false_positive + strange == total
    acc = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    print('acc', acc, 'precision', precision, 'recall', recall, 'f1', f1, sep='\t')
    
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
    _data = data_processing.DataProcessing()
    bayes = bayesline.BayesClassifier()
    FEATURES = [_data.num_dashes,
                _data.avg_word_length,
                _data.num_words,
                _data.num_possessive,
                _data.num_commas,
                _data.num_colons,
                _data.num_semicolons,
                _data.ampersands,
                _data.dollars,
                _data.percents,
                _data.parens,
                _data.question,
                _data.exclamation,
                _data.frequent_words,
            ]
    perceptron = Perceptron(FEATURES, _data)
    
    

    baseline_vocab = _data.get_word_counts_by_label(_data.training_set)
    print("Word Counts Baseline Validation Set")
    evaluate_baseline(_data.validate_set, baseline_vocab)
    print("Word Counts Baseline Test Set")
    evaluate_baseline(_data.test_set, baseline_vocab)
    
    bayes_valid = bayes.bayes_classify(_data.validate_set)
    bayes_test = bayes.bayes_classify(_data.test_set)
    print()
    perceptron.fit_decision_tree()
    tvalid_preds, ttest_preds, tvalid_labels, ttest_labels = perceptron.classify_decision_tree()
    print('Tree Validation Metrics:')
    eval_metrics(tvalid_labels, tvalid_preds)
    print('Tree Test Metrics:')
    eval_metrics(ttest_labels, ttest_preds)

    print()
    perceptron.fit_knn()
    kvalid_preds, ktest_preds, kvalid_labels, ktest_labels  = perceptron.classify_knn()
    print('KNN Validation Metrics:')
    eval_metrics(kvalid_labels, kvalid_preds)
    print('KNN Test Metrics:')
    eval_metrics(ktest_labels, ktest_preds)
    print()
    for i in range(0,15):
        print("Epoch", i)
        perceptron.train(1)
        print('Perceptron Validation Set:')
        valid_perceptron_preds = perceptron.predict(_data.validate_set)
        eval_metrics(kvalid_labels, valid_perceptron_preds)
        print('Perceptron Test Set')
        test_perceptron_preds = perceptron.predict(_data.test_set)
        eval_metrics(ktest_labels, test_perceptron_preds)
        print()
        print('Majority Vote Validation Set:')
        majority_val_preds = perceptron.majority_vote_prediction(valid_perceptron_preds, kvalid_preds, tvalid_preds)
        eval_metrics(kvalid_labels, majority_val_preds)
        print('Majority Vote Test Set:')
        majority_test_preds = perceptron.majority_vote_prediction(test_perceptron_preds, ktest_preds, ttest_preds)
        eval_metrics(ktest_labels, majority_test_preds)
        print('\n\n')

    
    print('done')