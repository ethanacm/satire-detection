

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


