"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import logging
import os

# User imports.
from . import normalise
from Utilities import variable_indices_from_config

# Globals.
LOGGER = logging.getLogger(__name__)


def main(fileExamples, dirOutput, config, fileTargets=None):
    """Shard the dataset of examples.

    :param fileExamples:    The location of the file containing the dataset of input examples.
    :type fileExamples:     str
    :param dirOutput:       The location of the directory to write out the sharded files to.
    :type dirOutput:        str
    :param config:          The object containing the configuration parameters for the sharding.
    :type config:           Configuration.Configuration
    :param fileTargets:     The location of the file containing the targets of the input examples.
    :type fileTargets:      str

    """

    # =============================================== #
    # Divide the Data into Train, Test and Validation #
    # =============================================== #
    fileTrainData = os.path.join(dirOutput, "TempTrainData")
    fileTestData = os.path.join(dirOutput, "TempTestData")
    fileValidationData = os.path.join(dirOutput, "TempValidationData")
    with open(fileExamples, 'r'):
        # Extract the header (if one is present) and determine each variable's index. Also calculate the number of
        # variables in the dataset.
        header = {}
        separator = config.get_param(["DataPreparation", "DataProperties", "Separator"])[1]
        firstLine = open(fileExamples, 'r').readline().split(separator)
        numVariables = len(firstLine)
        if config.get_param(["DataPreparation", "DataProperties", "HeaderPresent"])[1]:
            header = {j: i for i, j in enumerate(firstLine)}

        # Determine the fraction of examples to go in each of the train, test and validation splits. Pad the
        # configuration parameters with 0s so that missing test and validation fraction values mean that there are no
        # examples allocated to those splits.
        datasetDivisions = config.get_param(["DataPreparation", "DataSplit"])[1]
        datasetDivisions[len(datasetDivisions):3] = [0] * (3 - len(datasetDivisions))  # Pad with 0s.
        trainFraction = datasetDivisions[0]
        testFraction = min(1 - trainFraction, datasetDivisions[1])
        validationFraction = min(1 - (trainFraction + testFraction), datasetDivisions[2])

        # Split the dataset.

    # ========================================= #
    # Determine Variables Needing Normalisation #
    # ========================================= #
    varsOneOfC = set()
    varsOneOfCMin1 = set()
    varsMinMax = set()
    varsStandardise = set()
    if config.get_param(["DataPreparation", "Normalise"])[0]:
        # Some variables are supposed to be normalised.
        LOGGER.info("Now determining how to normalise variables.")

        # Determine categorical normalisations needed.
        categoricalNormalising = config.get_param(["DataPreparation", "Normalise", "Categorical"])
        if categoricalNormalising[0]:
            if categoricalNormalising[1].get("OneOfC"):
                # Determine variables needing one-of-C normalisation.
                varsOneOfC = variable_indices_from_config.main(
                    categoricalNormalising[1]["OneOfC"]["NumericIndices"],
                    categoricalNormalising[1]["OneOfC"]["VariableNames"],
                    numVariables, header
                )
            if categoricalNormalising[1].get("OneOfC-1"):
                # Determine variables needing one-of-C-1 normalisation.
                varsOneOfCMin1 = variable_indices_from_config.main(
                    categoricalNormalising[1]["OneOfC-1"]["NumericIndices"],
                    categoricalNormalising[1]["OneOfC-1"]["VariableNames"],
                    numVariables, header
                )

        # Determine numeric normalisations needed.
        numericNormalising = config.get_param(["DataPreparation", "Normalise", "Numeric"])
        if numericNormalising[0]:
            if numericNormalising[1].get("MinMaxScale"):
                # Determine variables needing min-max normalisation.
                varsMinMax = variable_indices_from_config.main(
                    numericNormalising[1]["MinMaxScale"]["NumericIndices"],
                    numericNormalising[1]["MinMaxScale"]["VariableNames"],
                    numVariables, header
                )
            if numericNormalising[1].get("Standardise"):
                # Determine variables needing standardising.
                varsStandardise = variable_indices_from_config.main(
                    numericNormalising[1]["Standardise"]["NumericIndices"],
                    numericNormalising[1]["Standardise"]["VariableNames"],
                    numVariables, header
                )

    # ================================= #
    # Determine Normalisation Functions #
    # ================================= #





    # Determine the variables to ignore.
    varsToIgnore = config.get_param(["DataPreparation", "DataProperties", "VariablesToIgnore"])
    if varsToIgnore[1]:
        # Variables are supposed to be ignored.
        LOGGER.info("Now extracting the indices of the variables to ignore.")
        varsToIgnore = variable_indices_from_config.main(
            varsToIgnore[1]["NumericIndices"], varsToIgnore[1]["VariableNames"], numVariables, header
        )
    else:
        # No variables are being ignored.
        varsToIgnore = set()
    exampleID = config.get_param(["DataPreparation", "DataProperties", "ExampleIDVariable"])
    if exampleID[0]:
        # If there is an ID for each example, then that 'variable' should be ignored as well. As the column that the
        # ID is in could be 0 (a Falsey value) we test for None.
        varsToIgnore.add(exampleID[1])
