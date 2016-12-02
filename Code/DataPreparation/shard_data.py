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

    # ================== #
    # Normalise the Data #
    # ================== #

