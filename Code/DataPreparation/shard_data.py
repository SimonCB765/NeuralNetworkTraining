"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import logging
import os
import random
import sys

# User imports.
from . import normalise
from Utilities import variable_indices_from_config

# Globals.
LOGGER = logging.getLogger(__name__)

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    from itertools import zip_longest as izip_longest
else:
    from itertools import izip_longest


def shard_sequence(fileExamples, dirOutput, config, fileTargets=None):
    """Shard a dataset where each example is a sequence.

    :param fileExamples:    The location of the file containing the dataset of input examples.
    :type fileExamples:     str
    :param dirOutput:       The location of the directory to write out the sharded files to.
    :type dirOutput:        str
    :param config:          The object containing the configuration parameters for the sharding.
    :type config:           Configuration.Configuration
    :param fileTargets:     The location of the file containing the targets of the input examples.
    :type fileTargets:      str

    """

    pass


def shard_vector(fileExamples, dirOutput, config, fileTargets=None):
    """Shard a dataset where each example is a single vector.

    :param fileExamples:    The location of the file containing the dataset of input examples.
    :type fileExamples:     str
    :param dirOutput:       The location of the directory to write out the sharded files to.
    :type dirOutput:        str
    :param config:          The object containing the configuration parameters for the sharding.
    :type config:           Configuration.Configuration
    :param fileTargets:     The location of the file containing the targets of the input examples.
    :type fileTargets:      str

    """

    # Seed the random number generator.
    randomSeed = config.get_param(["RandomSeed"])
    if randomSeed[0]:
        random.seed(randomSeed[1])
    else:
        randomSeed.seed(None)

    dataType = config.get_param(["DataType"])  # Extract the the type of the data.

    # Extract the header (if one is present) and determine each variable's index. Also calculate the number of
    # variables in the dataset.
    headersPresent = config.get_param(["DataPreparation", "DataProperties", "HeaderPresent"])[1]
    separator = config.get_param(["DataPreparation", "DataProperties", "Separator"])[1]  # Extract the separator string.
    firstLine = open(fileExamples, 'r').readline().split(separator)
    header = {}
    if headersPresent:
        header = {j: i for i, j in enumerate(firstLine)}
    numVariables = len(firstLine)  # Number of variables (including the ID variable if there is one).

    # ========================================= #
    # Determine Variables Needing Normalisation #
    # ========================================= #
    varsOneOfC = {}
    varsOneOfCMin1 = {}
    varsMinMax = {}
    varsStandardise = {}
    if config.get_param(["DataPreparation", "Normalise"])[0]:
        # Some variables are supposed to be normalised.
        LOGGER.info("Now determining how to normalise variables.")

        # Determine categorical normalisations needed.
        categoricalNormalising = config.get_param(["DataPreparation", "Normalise", "Categorical"])
        if categoricalNormalising[0]:
            if categoricalNormalising[1].get("OneOfC"):
                # Determine variables needing one-of-C normalisation and initialise category information.
                varsOneOfC = variable_indices_from_config.main(
                    categoricalNormalising[1]["OneOfC"]["NumericIndices"],
                    categoricalNormalising[1]["OneOfC"]["VariableNames"],
                    numVariables, header
                )
                varsOneOfC = {i: set() for i in varsOneOfC}
            if categoricalNormalising[1].get("OneOfC-1"):
                # Determine variables needing one-of-C-1 normalisation and initialise category information..
                varsOneOfCMin1 = variable_indices_from_config.main(
                    categoricalNormalising[1]["OneOfC-1"]["NumericIndices"],
                    categoricalNormalising[1]["OneOfC-1"]["VariableNames"],
                    numVariables, header
                )
                varsOneOfCMin1 = {i: set() for i in varsOneOfCMin1}

        # Determine numeric normalisations needed.
        numericNormalising = config.get_param(["DataPreparation", "Normalise", "Numeric"])
        if numericNormalising[0]:
            if numericNormalising[1].get("MinMaxScale"):
                # Determine variables needing min-max normalisation and initialise min and max.
                varsMinMax = variable_indices_from_config.main(
                    numericNormalising[1]["MinMaxScale"]["NumericIndices"],
                    numericNormalising[1]["MinMaxScale"]["VariableNames"],
                    numVariables, header
                )
                varsMinMax = {i: {"Min": sys.maxsize, "Max": -sys.maxsize} for i in varsMinMax}
            if numericNormalising[1].get("Standardise"):
                # Determine variables needing standardising and initialise mean, number of examples and squares of
                # differences from the current mean.
                varsStandardise = variable_indices_from_config.main(
                    numericNormalising[1]["Standardise"]["NumericIndices"],
                    numericNormalising[1]["Standardise"]["VariableNames"],
                    numVariables, header
                )
                varsStandardise = {i: {"Num": 0, "Mean": 0.0, "SumDiffs": 0.0} for i in varsStandardise}

    # ===================================================== #
    # Divide the Data and Determine Normalisation Functions #
    # ===================================================== #
    dirTrainExamples = os.path.join(dirOutput, "TrainingData")
    os.makedirs(dirTrainExamples)
    fileTestExamples = os.path.join(dirOutput, "TestExamples")
    fileTestTargets = os.path.join(dirOutput, "TestTargets")
    fileValExamples = os.path.join(dirOutput, "ValidationExamples")
    fileValTargets = os.path.join(dirOutput, "ValidationTargets")
    with open(fileExamples, 'r') as fidExamples, open(fileTargets if fileTargets else os.devnull, 'r') as fidTargets, \
            open(fileTestExamples, 'w') as fidTestExamples, open(fileTestTargets, 'w') as fidTestTargets, \
            open(fileValExamples, 'w') as fidValExamples, open(fileValTargets, 'w') as fidValTargets:
        # Strip the header if one is present.
        if headersPresent:
            fidExamples.readline()
            fidTargets.readline()

        # Setup the training set shards.
        examplesPerShard = config.get_param(["DataPreparation", "ExamplesPerShard"])[1]  # Examples to put in a shard.
        examplesAddedToShard = 0  # The number of examples added to the current shard.
        currentFileNumber = 0
        fidExampleShard = open(os.path.join(dirTrainExamples, "Shard_{:d}_Example.txt".format(currentFileNumber)), 'w')
        fidTargetShard = open(os.path.join(dirTrainExamples, "Shard_{:d}_Target.txt".format(currentFileNumber)), 'w')

        # Determine the fraction of examples to go in each of the train, test and validation splits. Pad the
        # configuration parameters with 0s so that missing test and validation fraction values mean that there are no
        # examples allocated to those splits.
        datasetDivisions = config.get_param(["DataPreparation", "DataSplit"])[1]
        datasetDivisions[len(datasetDivisions):3] = [0] * (3 - len(datasetDivisions))  # Pad with 0s.
        trainFraction = datasetDivisions[0]
        testFraction = min(1 - trainFraction, datasetDivisions[1])
        validationFraction = min(1 - (trainFraction + testFraction), datasetDivisions[2])
        choices = [trainFraction, trainFraction + testFraction, trainFraction + testFraction + validationFraction]

        # Split the dataset.
        for example, target in izip_longest(fidExamples, fidTargets, fillvalue=''):
            exVars = example.split(separator)
            targetVars = target.split(separator) if target else []

            # Update categories for categorical variables.
            for i in varsOneOfC:
                varsOneOfC[i].add(exVars[i])
            for i in varsOneOfCMin1:
                varsOneOfCMin1[i].add(exVars[i])

            # Determine which direction this example should go.
            choice = random.random()
            choice = [choice < i for i in choices]
            if choice[0]:
                # The example will go to the training set.
                fidExampleShard.write(example)
                fidTargetShard.write(target)
                examplesAddedToShard += 1

                # Update numeric normalisation parameters.
                for i in varsMinMax:
                    varsMinMax[i]["Max"] = max(varsMinMax[i]["Max"], float(exVars[i]))
                    varsMinMax[i]["Min"] = min(varsMinMax[i]["Min"], float(exVars[i]))
                for i in varsStandardise:
                    varsStandardise[i]["Num"] += 1
                    delta = float(exVars[i]) - varsStandardise[i]["Mean"]
                    varsStandardise[i]["Mean"] += delta / varsStandardise[i]["Num"]
                    varsStandardise[i]["SumDiffs"] += delta * (float(exVars[i]) - varsStandardise[i]["Mean"])

                # Open a new shard file if needed.
                if examplesAddedToShard == examplesPerShard:
                    fidExampleShard.close()
                    fidTargetShard.close()
                    examplesAddedToShard = 0
                    currentFileNumber += 1
                    fidExampleShard = open(
                        os.path.join(dirTrainExamples, "Shard_{:d}_Example.txt".format(currentFileNumber)), 'w'
                    )
                    fidTargetShard = open(
                        os.path.join(dirTrainExamples, "Shard_{:d}_Target.txt".format(currentFileNumber)), 'w'
                    )
            elif choice[1]:
                # The example will go to the test set.
                fidTestExamples.write(example)
                fidTestTargets.write(target)
            elif choice[2]:
                # The example will go to the validation set.
                fidValExamples.write(example)
                fidValTargets.write(target)
            else:
                # The example will not go to any of the sets.
                pass
    # Close the final shard files.
    fidExampleShard.close()
    fidTargetShard.close()

    # ================================= #
    # Determine the Variables to Ignore #
    # ================================= #
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
        # If there is an ID for each example, then that 'variable' should be ignored as well.
        varsToIgnore.add(exampleID[1])
