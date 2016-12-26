"""Class for implementing normalisation techniques.

Records parameters needed to normalise individual variables and implements normalisation functions.

"""

# Python imports.
import re
import sys

# User imports.
from . import Normalisers

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    basestring = unicode = str


class BaseNormaliser(object):
    """Class for performing normalisation when there is no need to normalise anything."""

    def normalise(self, datapoint):
        """Normalise a datapoint's values.

        :param datapoint:   The datapoint needing its values normalised.
        :type datapoint:    list
        :return:            The normalised datapoint.
        :rtype:             list

        """

        return datapoint


class DataNormalisation(BaseNormaliser):
    """Class for normalising datasets."""

    def __init__(self, fileDataset, config, dataPurpose="Examples"):
        """Initialise a Normaliser object

        :param fileDataset:     The location of the file containing the dataset of to normalise.
        :type fileDataset:      str
        :param config:          The object containing the configuration parameters for the sharding.
        :type config:           JsonschemaManipulation.Configuration
        :param dataPurpose:     The type of data in the dataset (either "Examples" or "Targets").
        :type dataPurpose:      str

        """

        super(DataNormalisation, self).__init__()

        # Set whether examples or targets are being extracted.
        self._dataPurpose = dataPurpose

        # Extract properties of the dataset.
        self._separator = config.get_param(["DataProcessing", self._dataPurpose, "Separator"])[1]
        self._headerPresent = config.get_param(["DataProcessing", self._dataPurpose, "HeaderPresent"])[1]

        # Setup the header.
        with open(fileDataset, 'r') as fidDataset:
            line = fidDataset.readline()
            self._header = (line.strip()).split(self._separator)
            if not self._headerPresent:
                # Create a dummy header where each variable name is just the index at which it appears in the dataset.
                self._header = ["{:d}".format(i) for i in range(len(self._header))]

        # Extract the variables to ignore.
        varsToIgnore = config.get_param(["DataProcessing", self._dataPurpose, "VariablesToIgnore"])
        varsToIgnore = varsToIgnore[1] if varsToIgnore[0] else []
        self._varsToIgnore = self.determine_variable_names(varsToIgnore)
        idVariable = config.get_param(["DataProcessing", self._dataPurpose, "IDVariable"])
        self._idPresent = idVariable[0]
        if self._idPresent:
            # Ignore the ID variable if there is one.
            try:
                # Try adding the ID variable as if it is an integer.
                self._varsToIgnore.add(self._header[idVariable[1]])
            except TypeError:
                # The ID variable must be a string, so add it directly.
                self._varsToIgnore.add(idVariable[1])

        # Extract the variables to perform min max scaling on.
        minMaxNormVars = config.get_param(["DataProcessing", self._dataPurpose, "Normalise", "MinMaxScale"])
        minMaxNormVars = minMaxNormVars[1] if minMaxNormVars[0] else []
        self._minMaxNormVars = self.determine_variable_names(minMaxNormVars) - self._varsToIgnore

        # Extract the variables to perform standardisation on.
        standardiseVars = config.get_param(["DataProcessing", self._dataPurpose, "Normalise", "Standardise"])
        standardiseVars = standardiseVars[1] if standardiseVars[0] else []
        self._standardiseVars = self.determine_variable_names(standardiseVars) - self._varsToIgnore

        # Extract the variables to perform 1-of-C normalisation on.
        oneOfCVars = config.get_param(["DataProcessing", self._dataPurpose, "Normalise", "OneOfC"])
        oneOfCVars = oneOfCVars[1] if oneOfCVars[0] else []
        self._oneOfCVars = self.determine_variable_names(oneOfCVars) - self._varsToIgnore

        # Extract the variables to perform 1-of-(C-1) normalisation on.
        oneOfCMin1Vars = config.get_param(["DataProcessing", self._dataPurpose, "Normalise", "OneOfC-1"])
        oneOfCMin1Vars = oneOfCMin1Vars[1] if oneOfCMin1Vars[0] else []
        self._oneOfCMin1Vars = self.determine_variable_names(oneOfCMin1Vars) - self._varsToIgnore

    def determine_variable_names(self, refList):
        """Determine the names of the variables specified in a mixed list of regular expressions and numeric indices.

        :param refList: The references to extract variable names from. These references are a combination of
                        integer indices and strings containing regular expressions indicating the variable names.
        :type refList:  list
        :return:        The names specified in the list of references.
        :rtype:         set

        """

        variablesNames = set()

        # Split the references into numeric and variable name references.
        nameRegexps = set()
        for i in refList:
            # If i is an integer, then get the name of the variable. If i is not an integer, then it is a regexp
            # representing the name(s) of the variables.
            nameRegexps.add(i) if isinstance(i, basestring) else variablesNames.add(self._header[i])

        # Determine variable names from regular expressions. The given expressions are matched starting from the first
        # character in the variable name and ending at the end of the name, rather than being matched anywhere in it.
        # This ensures that something like Var_1 will match only Var_1 and not Var_11 etc.
        if nameRegexps:
            # Only bother if there are some names given.
            regex = re.compile('|'.join(["{:s}$".format(i) for i in nameRegexps]))
            variablesNames.update({i for i in self._header if regex.match(i)})

        return variablesNames


class BOWNormaliser(DataNormalisation):
    """Class for normalising bag-of-words datasets."""

    def __init__(self, fileDataset, config, dataPurpose=True):
        """Initialise a bag-of-words Normaliser object

        :param fileDataset:     The location of the file containing the dataset of to normalise.
        :type fileDataset:      str
        :param config:          The object containing the configuration parameters for the sharding.
        :type config:           JsonschemaManipulation.Configuration
        :param dataPurpose:     The type of data in the dataset (either "Examples" or "Targets").
        :type dataPurpose:      str

        """

        # Initialise the superclass.
        super(BOWNormaliser, self).__init__(fileDataset, config, dataPurpose)

        # Setup the normalisation classes.
        self._normalisers = {}
        baseNormaliser = Normalisers.BaseNormalisation()
        ignoreVarNorm = Normalisers.IgnoreVariable()
        for i in self._header:
            if i in self._varsToIgnore:
                self._normalisers[i] = ignoreVarNorm
            elif i in self._minMaxNormVars:
                self._normalisers[i] = Normalisers.MinMaxNorm()
            elif i in self._standardiseVars:
                self._normalisers[i] = Normalisers.Standardisation()
            elif i in self._oneOfCVars:
                self._normalisers[i] = Normalisers.OneOfC()
            elif i in self._oneOfCMin1Vars:
                self._normalisers[i] = Normalisers.OneOfCMin1()
            else:
                self._normalisers[i] = baseNormaliser

        # Setup the normaliser functions.
        with open(fileDataset, 'r') as fidDataset:
            # Strip the header if there is one.
            if self._headerPresent:
                fidDataset.readline()

            # Go through the datapoints and update the record for each normaliser function.
            for line in fidDataset:
                variables = (line.strip()).split(self._separator)
                for i in variables:
                    varName, varVal = i.split(':')
                    self._normalisers[varName].update(varVal)

        # Determine new variable indices.
        self._keptVariables = {}
        self._numVariables = 0
        for i in self._header:
            if i not in self._varsToIgnore:
                if i in self._oneOfCVars or i in self._oneOfCMin1Vars:
                    self._keptVariables[i] = []
                    for _ in range(self._normalisers[i].get_num_dummies()):
                        self._keptVariables[i].append(self._numVariables)
                        self._numVariables += 1
                else:
                    self._keptVariables[i] = [self._numVariables]
                    self._numVariables += 1

    def normalise(self, datapoint):
        """Normalise a datapoint's values.

        :param datapoint:   The datapoint needing its values normalised.
        :type datapoint:    list
        :return:            The normalised datapoint.
        :rtype:             list

        """

        normalisedDatapoint = []
        indices = []
        for i in datapoint:
            varName, varVal = i.split(':')
            varIndices = self._keptVariables.get(varName, [])
            indices.extend(varIndices)
            varNormVals = self._normalisers[varName].normalise(varVal)
            normalisedDatapoint.extend(varNormVals)

        return [self._numVariables, indices, normalisedDatapoint]


class VectorNormaliser(DataNormalisation):
    """Class for normalising vectors."""

    def __init__(self, fileDataset, config, dataPurpose=True):
        """Initialise a vector Normaliser object

        :param fileDataset:     The location of the file containing the dataset of to normalise.
        :type fileDataset:      str
        :param config:          The object containing the configuration parameters for the sharding.
        :type config:           JsonschemaManipulation.Configuration
        :param dataPurpose:     The type of data in the dataset (either "Examples" or "Targets").
        :type dataPurpose:      str

        """

        # Initialise the superclass.
        super(VectorNormaliser, self).__init__(fileDataset, config, dataPurpose)

        # Setup the normalisation classes.
        self._normalisers = {}
        baseNormaliser = Normalisers.BaseNormalisation()
        ignoreVarNorm = Normalisers.IgnoreVariable()
        for i, j in enumerate(self._header):
            if j in self._varsToIgnore:
                self._normalisers[i] = ignoreVarNorm
            elif j in self._minMaxNormVars:
                self._normalisers[i] = Normalisers.MinMaxNorm()
            elif j in self._standardiseVars:
                self._normalisers[i] = Normalisers.Standardisation()
            elif j in self._oneOfCVars:
                self._normalisers[i] = Normalisers.OneOfC()
            elif j in self._oneOfCMin1Vars:
                self._normalisers[i] = Normalisers.OneOfCMin1()
            else:
                self._normalisers[i] = baseNormaliser

        # Setup the normaliser functions.
        with open(fileDataset, 'r') as fidDataset:
            # Strip the header if there is one.
            if self._headerPresent:
                fidDataset.readline()

            # Go through the datapoints and update the record for each normaliser function.
            for line in fidDataset:
                for i, j in enumerate((line.strip()).split(self._separator)):
                    self._normalisers[i].update(j)

    def normalise(self, datapoint):
        """Normalise a datapoint's values.

        :param datapoint:   The datapoint needing its values normalised.
        :type datapoint:    list
        :return:            The normalised datapoint.
        :rtype:             list

        """

        normalisedDatapoint = []
        for i, j in enumerate(datapoint):
            normalisedDatapoint.extend(self._normalisers[i].normalise(j))

        return normalisedDatapoint
