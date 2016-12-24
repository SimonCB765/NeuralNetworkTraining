"""Classes for performing normalisations."""

# Python imports.
import math
import sys


class BaseNormalisation(object):
    """Base normalisation class."""

    def normalise(self, value):
        """Normalise a variable that is meant to be unchanged.

        :param value:   The value to normalise.
        :type value:    str
        :return:        The unnormalised input value.
        :rtype:         list

        """

        return [float(value)]

    def update(self, value):
        """Update the parameters used for the normalisation.

        :param value:   The value to use in the update.
        :type value:    str

        """

        pass


class IgnoreVariable(BaseNormalisation):
    """Class for ignoring a variable."""

    def normalise(self, value):
        """Normalise a variable that is meant to be ignored.

        :param value:   The value to normalise.
        :type value:    str
        :return:        An empty value.
        :rtype:         list

        """

        return []


class MinMaxNorm(BaseNormalisation):
    """Class for performing min max normalisation."""

    def __init__(self):
        """Initialise a min-max normaliser."""

        self._min = sys.maxsize
        self._max = -sys.maxsize

    def normalise(self, value):
        """Normalise a variable that is meant to be min-max normalised to the range [-1, 1].

        :param value:   The value to normalise.
        :type value:    str
        :return:        The normalised value.
        :rtype:         list

        """

        value = float(value)
        return [(value - ((self._max + self._min) / 2)) / ((self._max - self._min) / 2)]

    def update(self, value):
        """Update the parameters used for the normalisation.

        :param value:   The value to use in the update.
        :type value:    str

        """

        value = float(value)
        self._max = max(self._max, value)
        self._min = min(self._min, value)


class OneOfC(BaseNormalisation):
    """Class for performing one-of-C normalisation."""

    def __init__(self):
        """Initialise a one-of-C normaliser."""

        self._valueCount = 0
        self._valueMapping = {}

    def normalise(self, value):
        """Normalise a variable that is meant to be one-of-C normalised.

        :param value:   The value to normalise.
        :type value:    str
        :return:        The one-of-C normalised value.
        :rtype:         list

        """

        if self._valueCount <= 2:
            # If there are no more than two categories, then only return one value.
            return [1 if self._valueMapping[value] == 0 else -1]
        else:
            # If there are more than two categories, then return one value for each category.
            return [(1 if i == value else -1) for i in sorted(self._valueMapping)]

    def update(self, value):
        """Update the parameters used for the normalisation.

        :param value:   The value to use in the update.
        :type value:    str

        """

        if value not in self._valueMapping:
            self._valueMapping[value] = self._valueCount
            self._valueCount += 1


class OneOfCMin1(BaseNormalisation):
    """Class for performing one-of-(C-1) normalisation."""

    def __init__(self):
        """Initialise a one-of-(C-1) normaliser."""

        self._valueCount = 0
        self._valueMapping = {}

    def normalise(self, value):
        """Normalise a variable that is meant to be one-of-(C-1) normalised.

        :param value:   The value to normalise.
        :type value:    str
        :return:        The one-of-(C-1) normalised value.
        :rtype:         list

        """

        if self._valueCount <= 2:
            # If there are no more than two categories, then only return one value.
            return [1 if self._valueMapping[value] == 0 else -1]
        else:
            # If there are more than two categories, then return one less value than the number of categories.
            return [(1 if i == value else -1) for i in sorted(self._valueMapping)[:-1]]

    def update(self, value):
        """Update the parameters used for the normalisation.

        :param value:   The value to use in the update.
        :type value:    str

        """

        if value not in self._valueMapping:
            self._valueMapping[value] = self._valueCount
            self._valueCount += 1


class Standardisation(BaseNormalisation):
    """Class for performing standardisation."""

    def __init__(self):
        """Initialise a standardisation normaliser."""

        self._mean = 0.0
        self._num = 0
        self._sumDiffs = 0.0

    def normalise(self, value):
        """Normalise a variable that is meant to be standardised.

        :param value:   The value to normalise.
        :type value:    str
        :return:        The standardised value.
        :rtype:         list

        """

        value = float(value)
        variance = self._sumDiffs / (self._num - 1)
        return [(value - self._mean) / math.sqrt(variance)]

    def update(self, value):
        """Update the parameters used for the normalisation.

        :param value:   The value to use in the update.
        :type value:    str

        """

        value = float(value)
        self._num += 1
        delta = value - self._mean
        self._mean += delta / self._num
        self._sumDiffs += delta * (value - self._mean)
