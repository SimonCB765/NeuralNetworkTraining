"""Code to extract the indices of variables in a dataset from their specification in a configuration file."""

# Python imports.
import logging
import operator
import re
import sys

# Globals.
LOGGER = logging.getLogger(__name__)

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    iteritems = operator.methodcaller("items")
else:
    iteritems = operator.methodcaller("iteritems")


def main(numericIndices, variableNames, numVariables, mapNamesToIndices=None):
    """Determine the indices of a set of variables from their specification in the configuration file.

    Variables are defined in the configuration file using two lists:
        - A list of numeric indices. This list contains both individual indices (recorded as integers) and indices
            recorded as strings. The strings can be individual indices (e.g. "3") or ranges (e.g. ":3", "3:5" and "5:").
        - A list of variable names. This list contains regular expressions that identify the variables by name. The
            regular expressions are all intended to be matched from the start of the variable's name.

    :param numericIndices:      Numeric indices of the variables. Can contain strings and integers, with the strings
                                containing valid Python list ranges.
    :type numericIndices:       list[int | str]
    :param variableNames:       Regular expressions indicating the names of variables.
    :type variableNames:        list[str]
    :param numVariables:        The number of variables.
    :type numVariables:         int
    :param mapNamesToIndices:   A mapping from variable names to their indices. If a variable name is not in this
                                mapping, then it can be assumed to be an invalid name.
    :type mapNamesToIndices:    dict
    :return:                    The indices of the variables.
    :rtype:                     set

    """

    varIndices = set()  # The indices corresponding to the variables.

    # Determine numeric indices.
    for i, j in enumerate([str(j) for j in numericIndices]):
        indices = j.split(':')
        if len(indices) == 1:
            # The numeric index is a single value, so just add the index (provided it's an integer).
            try:
                varIndices.add(int(indices[0]))
            except ValueError:
                # The index wasn't an integer.
                LOGGER.warning("Numerical index {:s} (entry {:d}) is not a single integer as expected.".format(j, i))
        elif len(indices) == 2:
            # The numeric index is in the form start:, :stop or start:stop. Add the appropriate range of values
            # provided that all values are integers.
            try:
                if indices[0]:
                    # If there is a start index...
                    if indices[1]:
                        # and a stop index, then add the range between the two.
                        varIndices.update(range(int(indices[0]), int(indices[1])))
                    else:
                        # but no stop index, then add every index from the start to the end.
                        varIndices.update(range(int(indices[0]), numVariables))
                else:
                    # If there is no start index there must be a stop index. Add all indices from 0 to the stop index.
                    varIndices.update(range(0, int(indices[1])))
            except ValueError:
                # One of the indices wasn't an integer.
                LOGGER.warning(
                    "Numerical index {:s} (entry {:d}) is of the format start:stop, but contains a non-integer.".format(
                        j, i))
        elif len(indices) == 3:
            # The numeric index is in the form start:stop:step. Add the appropriate range of values provided that all
            # values are integers.
            try:
                if indices[0]:
                    # If there is a start index...
                    if indices[1]:
                        # and a stop index, then add the range between the two.
                        varIndices.update(range(int(indices[0]), int(indices[1]), int(indices[2])))
                    else:
                        # but no stop index, then add every index from the start to the end.
                        varIndices.update(range(int(indices[0]), numVariables, int(indices[2])))
                else:
                    # If there is no start index there must be a stop index. Add all indices from 0 to the stop index.
                    varIndices.update(range(0, int(indices[1]), int(indices[2])))
            except ValueError:
                # One of the indices wasn't an integer.
                LOGGER.warning(
                    "Numerical index {:s} (entry {:d}) is of the format start:stop:step, but contains a "
                    "non-ineger.".format(j, i))
        else:
            # There wee too many ':'.
            LOGGER.warning(
                "Numerical index {:s} (entry {:d}) is not formatted using Python format indexing.".format(j, i))

    # Determine indices from variable names. Given expressions are matched starting from the first character rather
    # than being matched anywhere in the variable's name.
    if mapNamesToIndices:
        # Only bother if there are some names given.
        regex = re.compile('|'.join(variableNames))  # Compiled regular expression pattern name1|name2|name3|...|nameN.
        varIndices.update([value for key, value in iteritems(mapNamesToIndices) if regex.match(key)])

    return varIndices
