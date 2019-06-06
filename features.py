



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
    return {'num_posessive': headline.count("'s")}

def num_hyphens(self, headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_hyphens': headline.count("-")}

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
    return {'num_commas': headline.count(":")}

def num_semicolons(self, headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_commas': headline.count(";")}

def avg_word_length(self, headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'avg_word_length':(len(headline) - headline.count(' '))/ len(headline.split())}







