"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import logging
import os
import random
import re
import sys

# 3rd party imports.
import dask.dataframe as dd
import tensorflow as tf

# Globals.
LOGGER = logging.getLogger(__name__)

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    basestring = unicode = str
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

    # Extract the example data.
    LOGGER.info("Now extracting and normalising example data.")
    exampleData = _normalise_data_matrix(fileExamples, config, True)
    exampleID = config.get_param(["DataProcessing", "Examples", "IDVariable"])
    if exampleID[0]:
        # Drop the ID from the example data if there is one.
        exampleData = exampleData.drop(str(exampleID[1]), axis=1)

    # Extract the target data.
    targetData = None
    if fileTargets:
        LOGGER.info("Now extracting and normalising target data.")
        targetData = _normalise_data_matrix(fileTargets, config, False)
        targetID = config.get_param(["DataProcessing", "Targets", "IDVariable"])
        if targetID[0]:
            # Drop the ID from the target data if there is one.
            targetData = targetData.drop(str(targetID[1]), axis=1)

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
    exampleIterator = exampleData.iterrows()
    targetIterator = targetData.iterrows() if targetData else []
    for i, j in izip_longest(exampleIterator, targetIterator, fillvalue=([], [])):
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
                        "Example": _float_feature(list(i[1])),
                        "Target": _float_feature(list(j[1]))
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
                        "Example": _float_feature(list(i[1])),
                        "Target": _float_feature(list(j[1]))
                    }
                )
            )
            fidTest.write(example.SerializeToString())

            pass
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
                        "Example": _float_feature(list(i[1])),
                        "Target": _float_feature(list(j[1]))
                    }
                )
            )
            fidValidation.write(example.SerializeToString())

            pass
        else:
            # The example/target will not go to any of the sets.
            pass

    # Close the final shard file.
    fidTrainingShard.close()
    fidTest.close()
    fidValidation.close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _determine_variable_names(refList, header):
        """Determine the names of the variables specified in a mixed list of regular expressions and numeric indices.

        :param refList: The references to extract variable indices for. These references are a combination of
                        integer indices and strings containing regular expressions indicating the variable
                        names to get indices for.
        :type refList:  list
        :param header:  The names of the variables in the order that they appear in the dataset (i.e. header[0] is the
                        first variable.
        :type header:   list
        :return:        The indices specified in the list of references.
        :rtype:         set

        """

        variablesNames = set()

        # Split the references into numeric and variable name references.
        nameIndices = set()
        for i in refList:
            nameIndices.add(i) if isinstance(i, basestring) else variablesNames.add(header[i])

        # Determine indices from variable names. The given expressions are matched starting from the first character
        # in the variable name rather than being matched anywhere in it.
        if nameIndices:
            # Only bother if there are some names given.
            regex = re.compile('|'.join(nameIndices))  # Regular expression pattern name1|name2|name3|...|nameN.
            variablesNames.add({i for i in header if regex.match(i)})

        return variablesNames


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _normalise_data_matrix(fileDataset, config, isExamples=True):
    """Normalise a dataset arranged as a matrix with one row per example and one column per variable.

    :param fileDataset:     The location of the file containing the dataset.
    :type fileDataset:      str
    :param config:          The object containing the configuration parameters for the sharding.
    :type config:           JsonschemaManipulation.Configuration
    :param isExamples:      Whether the dataset is a dataset of examples or their targets.
    :type isExamples:       bool
    :return:                The normalised dataset.
    :rtype:                 dask.DataFrame

    """

    # Set whether examples or targets are being extracted.
    dataPurpose = "Examples" if isExamples else "Targets"

    # Extract properties of the dataset.
    separator = config.get_param(["DataProcessing", dataPurpose, "Separator"])[1]
    headerPresent = config.get_param(["DataProcessing", dataPurpose, "HeaderPresent"])[1]

    # Load the data frame.
    dataset = dd.read_csv(fileDataset, sep=separator, header=0 if headerPresent else None)
    if not headerPresent:
        # If there is no header present, then set all the placeholder column headers to be strings. This enables
        # dataset.assig() to overwrite columns later.
        dataset = dataset.rename(columns={i: str(i) for i in dataset.columns.values})
    header = dataset.columns.values

    # Extract the variables to ignore.
    varsToIgnore = config.get_param(["DataProcessing", dataPurpose, "VariablesToIgnore"])
    varsToIgnore = varsToIgnore[1] if varsToIgnore[0] else []
    varsToIgnore = _determine_variable_names(varsToIgnore, header)

    # Mask out the variables to ignore.
    dataset = dataset.drop(varsToIgnore, axis=1)

    # Perform min max scaling.
    minMaxNormVars = config.get_param(["DataProcessing", dataPurpose, "Normalise", "MinMaxScale"])
    minMaxNormVars = minMaxNormVars[1] if minMaxNormVars[0] else []
    minMaxNormVars = list(_determine_variable_names(minMaxNormVars, header) - varsToIgnore)
    minVals = dataset[minMaxNormVars].min()
    maxVals = dataset[minMaxNormVars].max()
    columnNormalisations = {}
    for i, j in zip(minVals.iteritems(), maxVals.iteritems()):
        variable = i[0]
        minVal = i[1]
        maxVal = j[1]
        columnNormalisations[variable] = (dataset[variable] - ((maxVal + minVal) / 2)) / ((maxVal - minVal) / 2)
    dataset = dataset.assign(**columnNormalisations)

    # Perform standardisation.
    standardiseVars = config.get_param(["DataProcessing", dataPurpose, "Normalise", "Standardise"])
    standardiseVars = standardiseVars[1] if standardiseVars[0] else []
    standardiseVars = list(_determine_variable_names(standardiseVars, header) - varsToIgnore)
    meanVals = dataset[standardiseVars].mean()
    stdVals = dataset[standardiseVars].std()
    columnNormalisations = {}
    for i, j in zip(meanVals.iteritems(), stdVals.iteritems()):
        variable = i[0]
        meanVal = i[1]
        stdVal = j[1]
        columnNormalisations[variable] = (dataset[variable] - meanVal) / stdVal
    dataset = dataset.assign(**columnNormalisations)

    # Perform 1-of-C normalisation.
    oneOfCVars = config.get_param(["DataProcessing", dataPurpose, "Normalise", "OneOfC"])
    oneOfCVars = oneOfCVars[1] if oneOfCVars[0] else []
    oneOfCVars = _determine_variable_names(oneOfCVars, header) - varsToIgnore
    oneOfCNormalisations = {}
    dropVariables = []
    for i in oneOfCVars:
        # Create the dummy variables needed for each categorical variable. New columns/variables are only needed if
        # there are more than two categories.
        categories = list(dataset[i].unique().iteritems())
        if len(categories) > 2:
            dropVariables.append(i)
            for j in categories:
                dummyColumn = ((dataset[i] == j[1]) * 2) - 1
                oneOfCNormalisations["{:s}_{:s}".format(i, str(j[1]))] = dummyColumn
        else:
            # There's no need to create a new set of dummy variables, so just overwrite the existing variable.
            dummyColumn = ((dataset[i] == categories[0][1]) * 2) - 1
            oneOfCNormalisations[i] = dummyColumn
    dataset = dataset.assign(**oneOfCNormalisations)
    dataset = dataset.drop(dropVariables, axis=1)  # Drop the old categorical columns with more than two categories.

    # Perform 1-of-(C-1) normalisation.
    oneOfCMin1Vars = config.get_param(["DataProcessing", dataPurpose, "Normalise", "OneOfC-1"])
    oneOfCMin1Vars = oneOfCMin1Vars[1] if oneOfCMin1Vars[0] else []
    oneOfCMin1Vars = _determine_variable_names(oneOfCMin1Vars, header) - varsToIgnore
    oneOfCMin1Normalisations = {}
    dropVariables = []
    for i in oneOfCMin1Vars:
        # Create the dummy variables needed for each categorical variable. New columns/variables are only needed if
        # there are more than two categories.
        categories = list(dataset[i].unique().iteritems())
        if len(categories) > 2:
            dropVariables.append(i)
            for j in categories[:-1]:
                dummyColumn = ((dataset[i] == j[1]) * 2) - 1
                oneOfCMin1Normalisations["{:s}_{:s}".format(i, str(j[1]))] = dummyColumn
        else:
            # There's no need to create a new set of dummy variables, so just overwrite the existing variable.
            dummyColumn = ((dataset[i] == categories[0][1]) * 2) - 1
            oneOfCMin1Normalisations[i] = dummyColumn
    dataset = dataset.assign(**oneOfCMin1Normalisations)
    dataset = dataset.drop(dropVariables, axis=1)  # Drop the old categorical columns with more than two categories.

    # Update the dataset.
    dataset.compute()

    return dataset
