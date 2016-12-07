"""Class for implementing normalisation techniques.

Records parameters needed to normalise individual variables and implements normalisation functions.

"""

# Python imports.
import logging
import operator
import re
import sys

# Globals.
LOGGER = logging.getLogger(__name__)

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    basestring = unicode = str
    iteritems = operator.methodcaller("items")
else:
    iteritems = operator.methodcaller("iteritems")


class Normaliser(object):
    """Class for normalising datasets."""

    def __init__(self, exampleHeader, numVariables, targetHeader=None, numTargets=0, varsToIgnore=None,
                 exampleNormVars=None, targetNormVars=None):
        """Initialise a Normaliser object."""

        # Initialise dataset properties.
        self._exampleHeader = exampleHeader  # Mapping from example names to their index.
        self._numVariables = numVariables  # Number of variables (including the ID variable if there is one).
        self._targetHeader = targetHeader  # Mapping from target names to their index.
        self._numTargets = numTargets  # Number of target variables.

        # Determine the variables to ignore.
        self._varsToIgnore = self.determine_indices(varsToIgnore, self._exampleHeader)

        # Extract the example and target variables to normalise and initialise the normalisation parameters.
        self._exampleNormVars = {
            i: (self.determine_indices(exampleNormVars.get(i, []), self._exampleHeader) - self._varsToIgnore)
            for i in exampleNormVars
        }
        self._targetNormVars = {
            i: (self.determine_indices(targetNormVars.get(i, []), self._targetHeader) - self._varsToIgnore)
            for i in targetNormVars
        }
        self._exampleNormParams = {}
        self._targetNormParams = {}
        self.initialise_norm_params()

        print(self._exampleNormVars)

    @staticmethod
    def determine_indices(refList, indexMapping):
        """Determine the indices of the variables specified in a mixed list of names and numeric indices.

        :param refList:         The references to extract variable indices for. These references are a combination of
                                integer indices and strings containing regular expressions indicating the variable
                                names to get indices for.
        :type refList:          list
        :param indexMapping:    A mapping from each variable's name to the index of the variable (if a header was
                                present in the dataset).
        :type indexMapping:     dict
        :return:                The indices specified in the list of references.
        :rtype:                 set

        """

        # Split the references into numeric and variable name references.
        numericIndices = set()
        nameIndices = set()
        for i in refList:
            nameIndices.add(i) if isinstance(i, basestring) else numericIndices.add(i)

        # Determine indices from variable names. The given expressions are matched starting from the first character
        # in the variable name rather than being matched anywhere in it.
        if nameIndices:
            # Only bother if there are some names given.
            regex = re.compile('|'.join(nameIndices))  # Regular expression pattern name1|name2|name3|...|nameN.
            nameIndices = {value for key, value in iteritems(indexMapping) if regex.match(key)}

        return nameIndices | numericIndices

    def initialise_norm_params(self):
        """Initialise the normalisation parameters for the different types of normalisation."""

        # Initialise example normalisation parameters.
        self._exampleNormParams = {
            "OneOfC": {i: set() for i in self._exampleNormVars.get("OneOfC", [])},
            "OneOfC-1": {i: set() for i in self._exampleNormVars.get("OneOfC-1", [])},
            "MinMaxScale":
                {i: {"Min": sys.maxsize, "Max": -sys.maxsize} for i in self._exampleNormVars.get("MinMaxScale", [])},
            "Standardise":
                {i: {"Num": 0, "Mean": 0.0, "SumDiffs": 0.0} for i in self._exampleNormVars.get("Standardise", [])}
        }

        # Initialise target normalisation parameters.
        self._targetNormParams = {
            "OneOfC": {i: set() for i in self._targetNormVars.get("OneOfC", [])},
            "OneOfC-1": {i: set() for i in self._targetNormVars.get("OneOfC-1", [])},
            "MinMaxScale":
                {i: {"Min": sys.maxsize, "Max": -sys.maxsize} for i in self._targetNormVars.get("MinMaxScale", [])},
            "Standardise":
                {i: {"Num": 0, "Mean": 0.0, "SumDiffs": 0.0} for i in self._targetNormVars.get("Standardise", [])}
        }

    def update_norm_params(self, exampleVars, targetVars):
        """Update the target and example normalisation parameters with the data on a single example.

        :param exampleVars:     The instantiations of the example variables.
        :type exampleVars:      list
        :param targetVars:      The instantiations of the target variables for this example.
        :type targetVars:       list

        """

        # Update categories for categorical variables.
        for i, j in iteritems(self._exampleNormParams["OneOfC"]):
            j.add(exampleVars[i])
        for i, j in iteritems(self._targetNormParams["OneOfC"]):
            j.add(targetVars[i])
        for i, j in iteritems(self._exampleNormParams["OneOfC-1"]):
            j.add(exampleVars[i])
        for i, j in iteritems(self._targetNormParams["OneOfC-1"]):
            j.add(targetVars[i])

        # Update numeric normalisation parameters.
        for i, j in iteritems(self._exampleNormParams["MinMaxScale"]):
            j["Max"] = max(j["Max"], float(exampleVars[i]))
            j["Min"] = min(j["Min"], float(exampleVars[i]))
        for i, j in iteritems(self._targetNormParams["MinMaxScale"]):
            j["Max"] = max(j["Max"], float(targetVars[i]))
            j["Min"] = min(j["Min"], float(targetVars[i]))
        for i, j in iteritems(self._exampleNormParams["Standardise"]):
            j["Num"] += 1
            delta = float(exampleVars[i]) - j["Mean"]
            j["Mean"] += delta / j["Num"]
            j["SumDiffs"] += delta * (float(exampleVars[i]) - j["Mean"])
        for i, j in iteritems(self._targetNormParams["Standardise"]):
            j["Num"] += 1
            delta = float(targetVars[i]) - j["Mean"]
            j["Mean"] += delta / j["Num"]
            j["SumDiffs"] += delta * (float(targetVars[i]) - j["Mean"])


class SequenceNormaliser(Normaliser):
    """Class for normalising datasets where each example is a sequence."""


class VectorNormaliser(Normaliser):
    """Class for normalising datasets where each example consists of a single vector."""

    def __init__(self, exampleHeader, numVariables, targetHeader=None, numTargets=0, varsToIgnore=None,
                 exampleNormVars=None, targetNormVars=None):
        """Initialise a VectorNormaliser object."""

        # Initialise the super class.
        super(VectorNormaliser, self).__init__(
            exampleHeader, numVariables, targetHeader=targetHeader, numTargets=numTargets, varsToIgnore=varsToIgnore,
            exampleNormVars=exampleNormVars, targetNormVars=targetNormVars
        )
