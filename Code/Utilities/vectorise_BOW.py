"""Code to vectorise a bag-of-words."""


def main(data, numVars, formatter=None, fillValue=0, separator=','):
    """Convert a bag-of-words to a vector.

    The bag-of-words is represented in the following formats:
        iterable - ["ID_1:Val_1", "ID_2:Val_2", ..., "ID_n:Val_n"]
        string   - "ID_1:Val_1,ID_2:Val_2,...,ID_n:Val_n"
    The IDs (ID_*) are all expected to be integers indicating the position in the vector that the associated value
    should occur.

    :param data:            The data represented as a bag-of-words.
    :type data:             str | iterable
    :param numVars:         The number of variables in the vector.
    :type numVars:          int
    :param formatter:       The method to use to format the values supplied in the bag-of-words. Defaults to converting
                                all values to integers.
    :type formatter:        function
    :param fillValue:       The value to use to fill in values not supplied in the bag-of-words.
    :type fillValue:        numeric | str
    :param separator:       The separator to use to split he bag-of-words when it is a string.
    :type separator:        str
    :return:                ?????????????????????????????????????????????????????
    :rtype:                 ?????????????????????????????????????????????????????

    """

    # Setup the formatter.
    if formatter is None:
        formatter = _int_formatter

    # Process the data if it is a string.
    try:
        data = data.split(separator)
    except AttributeError:
        # The data was supplied as an iterable so doesn't have an attribute called split.
        pass

    # Create the vector.
    bow = [fillValue] * numVars
    for i in data:
        chunks = i.split(':')
        bow[int(chunks[0])] = formatter(chunks[1])

    return bow


def _float_formatter(stringVal):
    """Convert a string to a float.

    :param stringVal:   The string to convert to a float.
    :type stringVal:    str
    :return:            stringVal converted to a float.
    :rtype:             float
    """

    return float(stringVal)


def _int_formatter(stringVal):
    """Convert a string to an integer.

    :param stringVal:   The string to convert to an integer.
    :type stringVal:    str
    :return:            stringVal converted to an integer.
    :rtype:             int
    """

    return int(stringVal)
