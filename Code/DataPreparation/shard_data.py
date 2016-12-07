"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import logging
import os
import random
import shutil
import sys

# User imports.
from . import Normaliser

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
        random.seed(None)

    dataType = config.get_param(["DataType"])  # Extract the the type of the data.

    # Extract properties of the dataset needed.
    exampleHeaderPresent = config.get_param(["DataPreparation", "DataProperties", "ExampleHeaderPresent"])[1]
    targetHeaderPresent = config.get_param(["DataPreparation", "DataProperties", "TargetHeaderPresent"])[1]
    separator = config.get_param(["DataPreparation", "DataProperties", "Separator"])[1]  # Extract the separator string.

    # Extract the variables to ignore.
    varsToIgnore = config.get_param(["DataPreparation", "DataProperties", "VariablesToIgnore"])
    varsToIgnore = varsToIgnore[1] if varsToIgnore[0] else []
    exampleID = config.get_param(["DataPreparation", "DataProperties", "ExampleIDVariable"])
    if exampleID[0]:
        # If there is an ID for each example, then that 'variable' should be ignored as well.
        varsToIgnore.append(exampleID[1])

    # For both the examples and targets, extract the header (if one is present), determine the number of
    # variables/targets in the dataset (including an ID variable if one is present) and determine each
    # variable's/target's index.
    firstLine = open(fileExamples, 'r').readline().split(separator)
    exampleHeader = {}
    if exampleHeaderPresent:
        exampleHeader = {j: i for i, j in enumerate(firstLine)}
    numVariables = len(firstLine)

    targetHeader = {}
    numTargets = 0
    if fileTargets:
        firstLine = open(fileTargets, 'r').readline().split(separator)
        if targetHeaderPresent:
            targetHeader = {j: i for i, j in enumerate(firstLine)}
        numTargets = len(firstLine)

    # Setup the normaliser object.
    exampleNormVars = config.get_param(["DataPreparation", "NormaliseExamples"])
    targetNormVars = config.get_param(["DataPreparation", "NormaliseTargets"])
    normaliser = Normaliser.VectorNormaliser(
        exampleHeader, numVariables, targetHeader=targetHeader, numTargets=numTargets, varsToIgnore=varsToIgnore,
        exampleNormVars=exampleNormVars[1] if exampleNormVars[0] else None,
        targetNormVars=targetNormVars[1] if targetNormVars[0] else None
    )

    # Determine the examples that will be used for training, testing and validation. Pad the
    # configuration parameters with 0s so that missing test and validation fraction values mean that there are no
    # examples allocated to those splits.
    datasetDivisions = config.get_param(["DataPreparation", "DataSplit"])[1]
    datasetDivisions[len(datasetDivisions):3] = [0] * (3 - len(datasetDivisions))  # Pad with 0s.
    trainFraction = datasetDivisions[0]
    testFraction = min(1 - trainFraction, datasetDivisions[1])
    validationFraction = min(1 - (trainFraction + testFraction), datasetDivisions[2])
    choices = [trainFraction, trainFraction + testFraction, trainFraction + testFraction + validationFraction]

    # ====================================================== #
    # Divide the Data and Determine Normalisation Parameters #
    # ====================================================== #
    dirTempTrainData = os.path.join(dirOutput, "TempTrainingData")
    os.makedirs(dirTempTrainData)
    fileTempTestExamples = os.path.join(dirOutput, "TempTestExamples")
    fileTempTestTargets = os.path.join(dirOutput, "TempTestTargets")
    fileTempValExamples = os.path.join(dirOutput, "TempValidationExamples")
    fileTempValTargets = os.path.join(dirOutput, "TempValidationTargets")
    with open(fileExamples, 'r') as fidExamples, open(fileTargets if fileTargets else os.devnull, 'r') as fidTargets, \
            open(fileTempTestExamples, 'w') as fidTestExamples, open(fileTempTestTargets, 'w') as fidTestTargets, \
            open(fileTempValExamples, 'w') as fidValExamples, open(fileTempValTargets, 'w') as fidValTargets:
        # Strip header if they're present.
        if exampleHeaderPresent:
            fidExamples.readline()
        if targetHeaderPresent:
            fidTargets.readline()

        # Setup the training set shards.
        examplesPerShard = config.get_param(["DataPreparation", "ExamplesPerShard"])[1]  # Examples to put in a shard.
        examplesAddedToShard = 0  # The number of examples added to the current shard.
        currentFileNumber = 0
        fidExampleShard = open(
            os.path.join(dirTempTrainData, "Shard_{:d}_Example".format(currentFileNumber)), 'w'
        )
        fidTargetShard = open(
            os.path.join(dirTempTrainData, "Shard_{:d}_Target".format(currentFileNumber)), 'w'
        )

        # Split the dataset.
        for example, target in izip_longest(fidExamples, fidTargets, fillvalue=''):
            exampleVars = example.split(separator)
            targetVars = target.split(separator) if fileTargets else []

            # Determine which direction this example should go.
            choice = random.random()
            choice = [choice < i for i in choices]
            if choice[0]:
                # The example will go to the training set.
                fidExampleShard.write(example)
                fidTargetShard.write(target)
                examplesAddedToShard += 1

                # Update the normalisation parameters
                normaliser.update_norm_params(exampleVars, targetVars)

                # Open a new shard file if needed.
                if examplesAddedToShard == examplesPerShard:
                    fidExampleShard.close()
                    fidTargetShard.close()
                    examplesAddedToShard = 0
                    currentFileNumber += 1
                    fidExampleShard = open(
                        os.path.join(dirTempTrainData, "Shard_{:d}_Example".format(currentFileNumber)), 'w'
                    )
                    fidTargetShard = open(
                        os.path.join(dirTempTrainData, "Shard_{:d}_Target".format(currentFileNumber)), 'w'
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

    # ============================== #
    # Normalise the Data and Save it #
    # ============================== #
    # Normalise and save the training data.
    dirTrainData = os.path.join(dirOutput, "TrainingData")
    os.makedirs(dirTrainData)
    trainingShardExamples = [
        (os.path.join(dirTempTrainData, i), os.path.join(dirTrainData, i)) for i in os.listdir(dirTempTrainData)
        if "Example" in i
        ]
    trainingShardTargets = [
        (os.path.join(dirTempTrainData, i), os.path.join(dirTrainData, i)) for i in os.listdir(dirTempTrainData)
        if "Target" in i
        ]
    for fileTempShard, fileShard in trainingShardExamples:
        with open(fileTempShard, 'r') as fidTempShard, open(fileShard, 'w') as fidShard:
            for example in fidTempShard:
                exampleVars = example.strip().split(separator)
                normalisedExample = normaliser.normalise(exampleVars)
    for fileTempShard, fileShard in trainingShardTargets:
        with open(fileTempShard, 'r') as fidTempShard, open(fileShard, 'w') as fidShard:
            for target in fidTempShard:
                targetVars = target.strip().split(separator)
                normalisedTarget = normaliser.normalise(targetVars, False)
    shutil.rmtree(dirTempTrainData)

    # Normalise and save the test data.
    fileTestExamples = os.path.join(dirOutput, "TestExamples")
    fileTestTargets = os.path.join(dirOutput, "TestTargets")
    with open(fileTempTestExamples, 'r') as fidTempTestExamples, open(fileTestExamples, 'w') as fidTestExamples:
        for example in fidTempTestExamples:
            exampleVars = example.strip().split(separator)
            normalisedExample = normaliser.normalise(exampleVars)
    with open(fileTempTestTargets, 'r') as fidTempTestTargets, open(fileTestTargets, 'w') as fidTestTargets:
        for target in fidTempTestTargets:
            targetVars = target.strip().split(separator)
            normalisedTarget = normaliser.normalise(targetVars, False)
    os.remove(fileTempTestExamples)
    os.remove(fileTempTestTargets)

    # Normalise and save the validation data.
    fileValExamples = os.path.join(dirOutput, "ValidationExamples")
    fileValTargets = os.path.join(dirOutput, "ValidationTargets")
    with open(fileTempValExamples, 'r') as fidTempValExamples, open(fileValExamples, 'w') as fidValExamples:
        for example in fidTempValExamples:
            exampleVars = example.strip().split(separator)
            normalisedExample = normaliser.normalise(exampleVars)
    with open(fileTempValTargets, 'r') as fidTempValTargets, open(fileValTargets, 'w') as fidValTargets:
        for target in fidTempValTargets:
            targetVars = target.strip().split(separator)
            normalisedTarget = normaliser.normalise(targetVars, False)
    os.remove(fileTempValExamples)
    os.remove(fileTempValTargets)
