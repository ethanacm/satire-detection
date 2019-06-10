import json
import random
import string
import operator

LIMIT = 10000


class DataProcessing:

    def __init__(self, seed=123497):
        self.vocab = {}
        self.preprocessed_vocab = False
        self.topics = {}
        self.preprocessed_topics = False

        data = []
        with open('Sarcasm_Headlines_Dataset.json', 'r') as file:
            for line in file:
                data.append(json.loads(line))

        random.seed(seed)
        random.shuffle(data)
        self.training_set = data[:int(.8 * len(data))]
        self.get_vocab(self.training_set)
        self.validate_set = data[int(.8 * len(data)):int(.9 * len(data))]
        self.test_set = data[int(.9 * len(data)):]

    def get_vocab(self, headlines, limit=None):
        if not self.preprocessed_vocab:
            self.vocab = self.get_vocab_from_set(self.training_set, limit=LIMIT)
            self.preprocessed_vocab = True
        return self.vocab

    def get_vocab_from_set(self, headlines, limit=None):
        vocab = {}
        for headline in headlines:
            text = headline['headline'].replace('-', ' ').split()
            for word in text:
                table = str.maketrans(dict.fromkeys(string.punctuation))
                word = word.translate(table)
                try:
                    float(word)
                    if '#NUMBER#' in vocab:
                        vocab['#NUMBER#'] += 1
                    else:
                        vocab['#NUMBER#'] = 1
                except:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
        if limit is None:
            return set(vocab.keys())
        else:
            return_vocab = set()
            #print('items', vocab.items())
            sorted_dict = sorted(vocab.items(), key=lambda item: -item[1])
            #print('sorted', sorted_dict)
            for word in sorted_dict[:limit]:
                return_vocab.add(word[0])
            return_vocab.add('<UNK>')
            return return_vocab

    def get_word_counts_by_label(self, headlines):
        vocab = {1: {},
                 0: {}}
        for headline in headlines:
            label = headline['is_sarcastic']
            text = headline['headline'].replace('-', ' ').split()
            for word in text:
                table = str.maketrans(dict.fromkeys(string.punctuation))
                word = word.translate(table)
                try:
                    float(word)
                except:
                    if word in vocab[label]:
                        vocab[label][word] += 1
                    else:
                        vocab[label][word] = 1
        return vocab


    ######################### FEATURES #########################
    def num_dashes(self, headline):
        """
        :param headline: headline to parse
        :return: dict of features for that headline
        """
        return {'num_dashes': headline.count('-') + headline.count('–')}


    def num_words(self, headline):
        """
        :param headline: headline to parse
        :return: dict of features for that headline
        """
        return {'num_words': len(headline.split())}


    def num_possessive(self, headline):
        """
        :param headline: headline to parse
        :return: dict of features for that headline
        """
        return {'num_posessive': headline.count("'s") + headline.count("s'")}


    def num_commas(self, headline):
        """
        :param headline: headline to parse
        :return: dict of features for that headline
        """
        return {'num_commas': headline.count(",")}


    def num_colons(self, headline):
        """
        :param headline: headline to parse
        :return: dict of features for that headline
        """
        return {'num_colons': headline.count(":")}


    def num_semicolons(self, headline):
        """
        :param headline: headline to parse
        :return: dict of features for that headline
        """
        return {'num_semicolons': headline.count(";")}


    def avg_word_length(self, headline):
        """
        :param headline: headline to parse
        :return: dict of features for that headline
        """
        try:
            return {'avg_word_length': (len(headline) - headline.count(' ')) / len(headline.split())}
        except:
            return {'avg_word_length': 0}


    def frequent_words(self, headline):
        words_dict = dict.fromkeys(self.vocab, 0)
        words_dict['<UNK>'] = 0
        for word in headline.replace('-', ' ').split():
            table = str.maketrans(dict.fromkeys(string.punctuation))
            word = word.translate(table)
            try:
                words_dict[word] += 1
            except:
                words_dict['<UNK>'] += 1
        return words_dict
