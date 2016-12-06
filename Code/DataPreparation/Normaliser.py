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

    def __init__(self, fileExamples, exampleHeaderPresent=False, fileTargets=None, targetHeaderPresent=False,
                 separator='\t', varsToIgnore=None, exampleNormVars=None, targetNormVars=None):
        """Initialise a Normaliser object."""

        # Initialise dataset properties.
        self._fileExamples = fileExamples
        self._exampleHeaderPresent = exampleHeaderPresent
        self._exampleHeader = {}  # Mapping from example names to their index.
        self._numVariables = 0  # Number of variables (including the ID variable if there is one).
        self._fileTargets = fileTargets
        self._targetHeaderPresent = exampleHeaderPresent
        self._targetHeader = {}  # Mapping from target names to their index.
        self._numTargets = 0  # Number of target variables.
        self._separator = separator

        # For both the examples and targets, extract the header (if one is present), determine the number of
        # variables/targets in the dataset (including an ID variable if one is present) and determine each
        # variable's/target's index.
        firstLine = open(self._fileExamples, 'r').readline().split(self._separator)
        if self._exampleHeaderPresent:
            self._exampleHeader = {j: i for i, j in enumerate(firstLine)}
        self._numVariables = len(firstLine)

        if self._fileTargets:
            firstLine = open(self._fileTargets, 'r').readline().split(self._separator)
            if self._targetHeaderPresent:
                self._targetHeader = {j: i for i, j in enumerate(firstLine)}
            self._numTargets = len(firstLine)

        # Determine the variables to ignore.
        self._varsToIgnore = self.determine_indices(varsToIgnore, self._exampleHeader)

        # Extract the example variables to normalise.
        self._oneOfCParamsExample = self.determine_indices(exampleNormVars.get("OneOfC", []), self._exampleHeader)
        self._oneOfCParamsExample = {i: set() for i in self._oneOfCParamsExample}
        self._oneOfCMin1ParamsExample = self.determine_indices(exampleNormVars.get("OneOfC-1", []), self._exampleHeader)
        self._oneOfCParamsExample = {i: set() for i in self._oneOfCMin1ParamsExample}
        self._minMaxParamsExample = self.determine_indices(exampleNormVars.get("MinMaxScale", []), self._exampleHeader)
        self._minMaxParamsExample = {i: {"Min": sys.maxsize, "Max": -sys.maxsize} for i in self._minMaxParamsExample}
        self._standardiseParamsExample = self.determine_indices(
            exampleNormVars.get("Standardise", []), self._exampleHeader
        )
        self._standardiseParamsExample = \
            {i: {"Num": 0, "Mean": 0.0, "SumDiffs": 0.0} for i in self._standardiseParamsExample}

        # Extract the target variables to normalise.
        self._oneOfCParamsTarget = self.determine_indices(exampleNormVars.get("OneOfC", []), self._exampleHeader)
        self._oneOfCParamsTarget = {i: set() for i in self._oneOfCParamsTarget}
        self._oneOfCMin1ParamsTarget = self.determine_indices(exampleNormVars.get("OneOfC-1", []), self._exampleHeader)
        self._oneOfCParamsTarget = {i: set() for i in self._oneOfCMin1ParamsTarget}
        self._minMaxParamsTarget = self.determine_indices(exampleNormVars.get("MinMaxScale", []), self._exampleHeader)
        self._minMaxParamsTarget = {i: {"Min": sys.maxsize, "Max": -sys.maxsize} for i in self._minMaxParamsTarget}
        self._standardiseParamsTarget = self.determine_indices(
            exampleNormVars.get("Standardise", []), self._exampleHeader
        )
        self._standardiseParamsTarget = \
            {i: {"Num": 0, "Mean": 0.0, "SumDiffs": 0.0} for i in self._standardiseParamsTarget}

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


    def update_parameters(self):
        """Update the normalisation parameters."""
        pass

    def normalise(self, example):
        """Normalise an example."""
        pass


class SequenceNormaliser(Normaliser):
    """Class for normalising datasets where each example is a sequence."""


class VectorNormaliser(Normaliser):
    """Class for normalising datasets where each example consists of a single vector."""

    def __init__(self, fileExamples, exampleHeaderPresent=False, fileTargets=None, targetHeaderPresent=False,
                 separator='\t', varsToIgnore=None, exampleNormVars=None, targetNormVars=None):
        """Initialise a VectorNormaliser object."""

        # Initialise the super class.
        super(VectorNormaliser, self).__init__(
            fileExamples, exampleHeaderPresent=exampleHeaderPresent, fileTargets=fileTargets,
            targetHeaderPresent=targetHeaderPresent, separator=separator, varsToIgnore=varsToIgnore,
            exampleNormVars=exampleNormVars, targetNormVars=targetNormVars
        )
