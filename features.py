

def num_dashes(headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_dashes': headline.count('-') + headline.count('–')}

def num_words(headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_words': len(headline.split())}

def num_possessive(headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_posessive': headline.count("'s")}

def num_hyphens(headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_hyphens': headline.count("-")}

def num_commas(headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_commas': headline.count(",")}

def num_colons(headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_commas': headline.count(":")}

def num_semicolons(headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'num_commas': headline.count(";")}

def avg_word_length(headline):
    """
    :param headline: headline to parse
    :return: dict of features for that headline
    """
    return {'avg_word_length':(len(headline) - headline.count(' '))/ len(headline.split())}







