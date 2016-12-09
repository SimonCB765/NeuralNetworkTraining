"""Class for implementing normalisation techniques.

Records parameters needed to normalise individual variables and implements normalisation functions.

"""

# Python imports.
from collections import OrderedDict
import logging
import operator
import re
import sys

# 3rd party imports.
import numpy as np

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

        # Convert the variables to ignore to a list for numpy array indexing.
        self._varsToIgnore = list(self._varsToIgnore)

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
            "OneOfC": {i: OrderedDict() for i in self._exampleNormVars.get("OneOfC", [])},
            "OneOfC-1": {i: OrderedDict() for i in self._exampleNormVars.get("OneOfC-1", [])},
            "MinMaxScale":
                {i: {"Min": sys.maxsize, "Max": -sys.maxsize} for i in self._exampleNormVars.get("MinMaxScale", [])},
            "Standardise":
                {i: {"Num": 0, "Mean": 0.0, "SumDiffs": 0.0} for i in self._exampleNormVars.get("Standardise", [])}
        }

        # Initialise target normalisation parameters.
        self._targetNormParams = {
            "OneOfC": {i: OrderedDict() for i in self._targetNormVars.get("OneOfC", [])},
            "OneOfC-1": {i: OrderedDict() for i in self._targetNormVars.get("OneOfC-1", [])},
            "MinMaxScale":
                {i: {"Min": sys.maxsize, "Max": -sys.maxsize} for i in self._targetNormVars.get("MinMaxScale", [])},
            "Standardise":
                {i: {"Num": 0, "Mean": 0.0, "SumDiffs": 0.0} for i in self._targetNormVars.get("Standardise", [])}
        }

    def normalise(self, vector, isExample=True):
        """Normalise an example or target vector.

        :param vector:      The example or target to normalise.
        :type vector:       list
        :param isExample:   Whether the vector is an example or target.
        :type isExample:    bool
        :return:            The normalised example or target.
        :rtype:             numpy.array

        """

        # Convert the non-normalised vector to a numpy array of strings.
        vector = np.array(vector)

        # Mark the variables to ignore as NaNs.
        vector[self._varsToIgnore] = np.NaN

        # Normalise categorical variables. This requires creating a new vector to represent the categorical value in
        # the vector, appending this to the end and then masking out the old categorical variable.
        if isExample:
            for i, j in iteritems(self._exampleNormParams["OneOfC"]):
                newCategories = np.full(len(j), -1)
                newCategories[list(j).index(vector[i])] = 1
                vector = np.append(vector, newCategories)
                vector[i] = np.NaN
            for i, j in iteritems(self._exampleNormParams["OneOfC-1"]):
                newCategories = np.full(len(j), -1)
                newCategories[list(j).index(vector[i])] = 1
                vector = np.append(vector, newCategories[:-1])
                vector[i] = np.NaN
        else:
            for i, j in iteritems(self._targetNormVars["OneOfC"]):
                newCategories = np.full(len(j), -1)
                newCategories[list(j).index(vector[i])] = 1
                vector = np.append(vector, newCategories)
                vector[i] = np.NaN
            for i, j in iteritems(self._targetNormVars["OneOfC-1"]):
                newCategories = np.full(len(j), -1)
                newCategories[list(j).index(vector[i])] = 1
                vector = np.append(vector, newCategories[:-1])
                vector[i] = np.NaN

        # Convert the array to floats.
        vector = vector.astype(np.float)

        # Perform numeric normalisation.
        if isExample:
            for i, j in iteritems(self._exampleNormParams["MinMaxScale"]):
                maxVal = self._exampleNormParams["MinMaxScale"][i]["Max"]
                minVal = self._exampleNormParams["MinMaxScale"][i]["Min"]
                vector[i] = (vector[i] - ((maxVal + minVal) / 2)) / ((maxVal - minVal) / 2)
            for i, j in iteritems(self._exampleNormParams["Standardise"]):
                mean = self._exampleNormParams["Standardise"][i]["Mean"]
                sumDiffs = self._exampleNormParams["Standardise"][i]["SumDiffs"]
                numExamples = self._exampleNormParams["Standardise"][i]["Num"]
                variance = sumDiffs / (numExamples - 1)
                vector[i] = (vector[i] - mean) / np.sqrt(variance)
        else:
            for i, j in iteritems(self._targetNormParams["MinMaxScale"]):
                maxVal = self._targetNormParams["MinMaxScale"][i]["Max"]
                minVal = self._targetNormParams["MinMaxScale"][i]["Min"]
                vector[i] = (vector[i] - ((maxVal + minVal) / 2)) / ((maxVal - minVal) / 2)
            for i, j in iteritems(self._targetNormParams["Standardise"]):
                mean = self._targetNormParams["Standardise"][i]["Mean"]
                sumDiffs = self._targetNormParams["Standardise"][i]["SumDiffs"]
                numExamples = self._targetNormParams["Standardise"][i]["Num"]
                variance = sumDiffs / (numExamples - 1)
                vector[i] = (vector[i] - mean) / np.sqrt(variance)

        # Remove all variables with value equal to NaN.
        vector = vector[~np.isnan(vector)]

        return vector

    def update_norm_params(self, exampleVars, targetVars):
        """Update the target and example normalisation parameters with the data on a single example.

        :param exampleVars:     The instantiations of the example variables.
        :type exampleVars:      list
        :param targetVars:      The instantiations of the target variables for this example.
        :type targetVars:       list

        """

        # Update categories for categorical variables.
        for i, j in iteritems(self._exampleNormParams["OneOfC"]):
            j[exampleVars[i]] = True
        for i, j in iteritems(self._targetNormParams["OneOfC"]):
            j[exampleVars[i]] = True
        for i, j in iteritems(self._exampleNormParams["OneOfC-1"]):
            j[exampleVars[i]] = True
        for i, j in iteritems(self._targetNormParams["OneOfC-1"]):
            j[exampleVars[i]] = True

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

    def update_norm_params(self, exampleVars, targetVars):
        """Update the target and example normalisation parameters with the data on a single example.

        :param exampleVars:     The instantiations of the example variables.
        :type exampleVars:      list
        :param targetVars:      The instantiations of the target variables for this example.
        :type targetVars:       list

        """

        super(VectorNormaliser, self).update_norm_params(exampleVars, targetVars)
