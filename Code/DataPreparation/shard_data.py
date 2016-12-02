"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import logging

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

    # If there is a header, then extract it and determine the index of each column.
    # Also determine the number of variables in the dataset.
    header = {}
    firstLine = open(fileExamples, 'r').readline().split(config.DataFormat["Separator"])
    numVariables = len(firstLine)
    if config.DataFormat["HeaderPresent"]:
        header = {j: i for i, j in enumerate(firstLine)}

    # Determine the variables to ignore.
    LOGGER.info("Now extracting the indices of the variables to ignore.")
    varsToIgnore = variable_indices_from_config.main(
        config.DataFormat["VariablesToIgnore"]["NumericIndices"],
        config.DataFormat["VariablesToIgnore"]["VariableNames"], numVariables, header
    )
    if config.DataFormat.get("ExampleIDVariable", None) is not None:
        # If there is an ID for each example, then that 'variable' should be ignored as well. As the column that the
        # ID is in could be 0 (a Falsey value) we test for None.
        varsToIgnore.add(config.DataFormat["ExampleIDVariable"])

    # ========================================= #
    # Determine Variables Needing Normalisation #
    # ========================================= #
    varsOneOfC = set()
    varsOneOfCMin1 = set()
    varsMinMax = set()
    varsStandardise = set()
    if config.ProcessData.get("Normalise"):
        # Some variables are supposed to be normalised.
        LOGGER.info("Now determining how to normalise variables.")

        # Determine categorical normalisations needed.
        if config.ProcessData["Normalise"].get("Categorical"):
            if config.ProcessData["Normalise"]["Categorical"].get("OneOfC"):
                # Determine variables needing one-of-C normalisation.
                varsOneOfC = variable_indices_from_config.main(
                    config.ProcessData["Normalise"]["Categorical"]["OneOfC"]["NumericIndices"],
                    config.ProcessData["Normalise"]["Categorical"]["OneOfC"]["VariableNames"],
                    numVariables, header
                )
            if config.ProcessData["Normalise"]["Categorical"].get("OneOfC-1"):
                # Determine variables needing one-of-C-1 normalisation.
                varsOneOfCMin1 = variable_indices_from_config.main(
                    config.ProcessData["Normalise"]["Categorical"]["OneOfC-1"]["NumericIndices"],
                    config.ProcessData["Normalise"]["Categorical"]["OneOfC-1"]["VariableNames"],
                    numVariables, header
                )

        # Determine numeric normalisations needed.
        if config.ProcessData["Normalise"].get("Numeric"):
            if config.ProcessData["Normalise"]["Numeric"].get("MinMaxScale"):
                # Determine variables needing min-max normalisation.
                varsMinMax = variable_indices_from_config.main(
                    config.ProcessData["Normalise"]["Numeric"]["MinMaxScale"]["NumericIndices"],
                    config.ProcessData["Normalise"]["Numeric"]["MinMaxScale"]["VariableNames"],
                    numVariables, header
                )
            if config.ProcessData["Normalise"]["Numeric"].get("Standardise"):
                # Determine variables needing standardising.
                varsStandardise = variable_indices_from_config.main(
                    config.ProcessData["Normalise"]["Numeric"]["Standardise"]["NumericIndices"],
                    config.ProcessData["Normalise"]["Numeric"]["Standardise"]["VariableNames"],
                    numVariables, header
                )
