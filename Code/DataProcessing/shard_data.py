"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import logging
import os
import random
import sys

# User imports.
from . import DataNormalisation

# 3rd party imports.
import tensorflow as tf

# Globals.
LOGGER = logging.getLogger(__name__)

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    from itertools import zip_longest as izip_longest
else:
    from itertools import izip_longest


def shard_image(fileExamples, dirOutput, config, fileTargets=None):
    """Shard a dataset where each example is an image.

    :param fileExamples:    The location of the file containing the dataset of input examples.
    :type fileExamples:     str
    :param dirOutput:       The location of the directory to write out the sharded files to.
    :type dirOutput:        str
    :param config:          The object containing the configuration parameters for the sharding.
    :type config:           JsonschemaManipulation.Configuration
    :param fileTargets:     The location of the file containing the targets of the input examples.
    :type fileTargets:      str

    """

    pass


def shard_sequence(fileExamples, dirOutput, config, fileTargets=None):
    """Shard a dataset where each example is a sequence.

    :param fileExamples:    The location of the file containing the dataset of input examples.
    :type fileExamples:     str
    :param dirOutput:       The location of the directory to write out the sharded files to.
    :type dirOutput:        str
    :param config:          The object containing the configuration parameters for the sharding.
    :type config:           JsonschemaManipulation.Configuration
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
    :type config:           JsonschemaManipulation.Configuration
    :param fileTargets:     The location of the file containing the targets of the input examples.
    :type fileTargets:      str

    """

    # Seed the random number generator.
    randomSeed = config.get_param(["RandomSeed"])
    if randomSeed[0]:
        random.seed(randomSeed[1])
    else:
        random.seed(None)

    # Determine the examples that will be used for training, testing and validation. Pad the
    # configuration parameters with 0s so that missing test and validation fraction values mean that there are no
    # examples allocated to those splits.
    datasetDivisions = config.get_param(["DataProcessing", "DataSplit"])[1]
    datasetDivisions[len(datasetDivisions):3] = [0] * (3 - len(datasetDivisions))  # Pad with 0s.
    trainFraction = datasetDivisions[0]
    testFraction = min(1 - trainFraction, datasetDivisions[1])
    validationFraction = min(1 - (trainFraction + testFraction), datasetDivisions[2])
    choices = [trainFraction, trainFraction + testFraction, trainFraction + testFraction + validationFraction]

    # Create the example data normaliser.
    LOGGER.info("Now creating example data normaliser.")
    isExamplesBOW = config.get_param(["ExampleBOW"])[1]
    if isExamplesBOW:
        exampleNormaliser = DataNormalisation.BOWNormaliser(fileExamples, config, dataPurpose="Examples")
    else:
        exampleNormaliser = DataNormalisation.VectorNormaliser(fileExamples, config, dataPurpose="Examples")

    # Create the target data normaliser.
    isTargetsBOW = config.get_param(["TargetBOW"])[1]
    if fileTargets:
        LOGGER.info("Now creating target data normaliser.")
        if isTargetsBOW:
            targetNormaliser = DataNormalisation.BOWNormaliser(fileTargets, config, dataPurpose="Targets")
        else:
            targetNormaliser = DataNormalisation.VectorNormaliser(fileTargets, config, dataPurpose="Targets")
    else:
        targetNormaliser = DataNormalisation.BaseNormaliser()

    # Setup the files to record the data in.
    dirTrainData = os.path.join(dirOutput, "TrainingData")
    os.makedirs(dirTrainData)
    examplesAddedToShard = 0  # The number of examples added to the current shard.
    currentFileNumber = 0
    fidTrainingShard = tf.python_io.TFRecordWriter(os.path.join(dirTrainData, "Shard_{:d}".format(currentFileNumber)))
    fidTest = tf.python_io.TFRecordWriter(os.path.join(dirOutput, "Test"))
    fidValidation = tf.python_io.TFRecordWriter(os.path.join(dirOutput, "Validation"))

    # Write out the examples and targets.
    LOGGER.info("Now writing out TFRecord files.")
    examplesPerShard = config.get_param(["DataProcessing", "ExamplesPerShard"])[1]  # Examples to put in a shard.
    exampleSeparator = config.get_param(["DataProcessing", "Examples", "Separator"])[1]
    exampleHeaderPresent = config.get_param(["DataProcessing", "Examples", "HeaderPresent"])[1]
    targetSeparator = config.get_param(["DataProcessing", "Targets", "Separator"])[1]
    targetHeaderPresent = config.get_param(["DataProcessing", "Targets", "HeaderPresent"])[1]
    with open(fileExamples, 'r') as fidExamples, open(fileTargets if fileTargets else os.devnull, 'r') as fidTargets:
        # Strip headers.
        if exampleHeaderPresent:
            fidExamples.readline()
        if targetHeaderPresent:
            fidTargets.readline()

        for i, j in izip_longest(fidExamples, fidTargets, fillvalue=''):
            # Normalise the data.
            exampleDatapoint = exampleNormaliser.normalise((i.strip()).split(exampleSeparator))
            targetDatapoint = [] if not fileTargets else targetNormaliser.normalise((j.strip()).split(targetSeparator))

            # Determine what dataset portion this example/target should go to.
            choice = random.random()
            choice = [choice < i for i in choices]
            if choice[0]:
                # The example/target will go to the training set.

                # Create the Example protocol buffer
                # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto).
                example = tf.train.Example(
                    # The Example protocol buffer contains a Features protocol buffer
                    # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto).
                    features=tf.train.Features(
                        # The Features protocol buffer contains a list of features, which are one of either a
                        # bytes_list, float_list or int64_list.
                        feature={
                            "Example": _bytes_feature(exampleDatapoint[1:]) if isExamplesBOW else \
                                _float_feature(exampleDatapoint),
                            "NumExampleVars": _int64_feature(
                                [exampleDatapoint[0]] if isExamplesBOW else [len(exampleDatapoint)]
                            ),
                            "NumTargetVars": _int64_feature(
                                [targetDatapoint[0]] if isTargetsBOW else [len(targetDatapoint)]
                            ),
                            "Target": _bytes_feature(targetDatapoint[1:]) if isTargetsBOW else \
                                _float_feature(targetDatapoint)
                        }
                    )
                )
                fidTrainingShard.write(example.SerializeToString())
                examplesAddedToShard += 1

                # Open a new shard file if needed.
                if examplesAddedToShard == examplesPerShard:
                    fidTrainingShard.close()
                    examplesAddedToShard = 0
                    currentFileNumber += 1
                    fidTrainingShard = tf.python_io.TFRecordWriter(
                        os.path.join(dirTrainData, "Shard_{:d}".format(currentFileNumber))
                    )
            elif choice[1]:
                # The example/target will go to the test set.

                # Create the Example protocol buffer
                # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto).
                example = tf.train.Example(
                    # The Example protocol buffer contains a Features protocol buffer
                    # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto).
                    features=tf.train.Features(
                        # The Features protocol buffer contains a list of features, which are one of either a
                        # bytes_list, float_list or int64_list.
                        feature={
                            "Example": _bytes_feature(exampleDatapoint[1:]) if isExamplesBOW else \
                                _float_feature(exampleDatapoint),
                            "NumExampleVars": _int64_feature(
                                [exampleDatapoint[0]] if isExamplesBOW else [len(exampleDatapoint)]
                            ),
                            "NumTargetVars": _int64_feature(
                                [targetDatapoint[0]] if isTargetsBOW else [len(targetDatapoint)]
                            ),
                            "Target": _bytes_feature(targetDatapoint[1:]) if isTargetsBOW else \
                                _float_feature(targetDatapoint)
                        }
                    )
                )
                fidTest.write(example.SerializeToString())
            elif choice[2]:
                # The example/target will go to the validation set.

                # Create the Example protocol buffer for the example
                # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto).
                example = tf.train.Example(
                    # The Example protocol buffer contains a Features protocol buffer
                    # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto).
                    features=tf.train.Features(
                        # The Features protocol buffer contains a list of features, which are one of either a
                        # bytes_list, float_list or int64_list.
                        feature={
                            "Example": _bytes_feature(exampleDatapoint[1:]) if isExamplesBOW else \
                                _float_feature(exampleDatapoint),
                            "NumExampleVars": _int64_feature(
                                [exampleDatapoint[0]] if isExamplesBOW else [len(exampleDatapoint)]
                            ),
                            "NumTargetVars": _int64_feature(
                                [targetDatapoint[0]] if isTargetsBOW else [len(targetDatapoint)]
                            ),
                            "Target": _bytes_feature(targetDatapoint[1:]) if isTargetsBOW else \
                                _float_feature(targetDatapoint)
                        }
                    )
                )
                fidValidation.write(example.SerializeToString())
            else:
                # The example/target will not go to any of the sets.
                pass

    # Close the final shard file.
    fidTrainingShard.close()
    fidTest.close()
    fidValidation.close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
