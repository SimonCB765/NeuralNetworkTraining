"""Code to setup a network that uses vector inputs."""

# User imports.
from Utilities import sparse_tensor_to_dense

# 3rd party imports.
import tensorflow as tf


def main(dirShardedFiles, config):
    """Create the input pipeline to read in examples and prepare them for training.

    :param dirShardedFiles:     The location of the directory containing the sharded files.
    :type dirShardedFiles:      str
    :param config:              The object containing the configuration parameters for the sharding.
    :type config:               JsonschemaManipulation.Configuration

    """

    # Determine whether the examples or targets are in bag-of-words format.
    isExamplesBOW = config.get_param(["ExampleBOW"])[1]
    isTargetsBOW = config.get_param(["TargetBOW"])[1]

    # Get the number of threads to use.
    numberThreads = config.get_param(["TensorflowParams", "NumberThreads"])[1]

    # Create the node that shuffles filenames and places them in a queue for consumption by the input pipeline.
    numberEpochs = config.get_param(["NetworkTraining", "NumberEpochs"])[1]
    randomSeed = config.get_param(["RandomSeed"])[1]
    trainingFiles = tf.train.match_filenames_once("{:s}/TrainingData/Shard_*".format(dirShardedFiles))
    filenameQueue = tf.train.string_input_producer(
        trainingFiles, num_epochs=numberEpochs, shuffle=True, seed=randomSeed
    )

    # Create the reader to read from the filename queue. In order to create mini-batches by reading from multiple files
    # we create a number of readers that share the same filename queue. This ensures that the different readers use
    # different files from the same epoch until all the files from the epoch have been started. Each reader returns a
    # record (key, value pair) from which we only want the value. As our batch joiner node expects a list of tuples
    # of tensors, we wrap each value tensor returned by a reader in a tuple (with the value tensor being the only
    # element. Each value tensor is of type string (i.e. it is a serialised example).
    readers = [tf.TFRecordReader() for _ in range(numberThreads)]
    serialisedExamples = [(i.read(filenameQueue)[1],) for i in readers]

    # Create the example batcher using shuffle_batch_join in order to use the examples read from multiple files (rather
    # than using shuffle_batch and generating each mini-batch from a single file). The batcher takes in the
    # serialised examples and makes batches of serialised examples available.
    # Batcher capacity must be larger than min_after_dequeue, and the amount larger determines the maximum number of
    # examples prefetched. Recommendation - min_after_dequeue + (num_threads + a small safety margin) * batch_size.
    # min_after_dequeue defines how big a buffer the batch will be randomly sampled from. Larger will give better
    # shuffling but slower start up and greater memory usage.
    batchSize = config.get_param(["NetworkTraining", "BatchSize"])[1]
    batcherMinAfterDequeue = config.get_param(["TensorflowParams", "BatcherMinAfterDequeue"])[1]
    batcherCapacity = config.get_param(["TensorflowParams", "BatcherCapacity"])[1]
    batcherCapacity = max(batcherCapacity, batcherMinAfterDequeue + (numberThreads + 2) * batchSize)
    batchedSerialisedExamples = tf.train.shuffle_batch_join(
        serialisedExamples, batchSize, capacity=batcherCapacity, min_after_dequeue=batcherMinAfterDequeue, seed=randomSeed
    )

    # Define the feature mapping used for decoding examples. As we don't know the number of variables in advance, we
    # use VarLenFeature for everything except the single integer indicating the number of variables.
    featureTypes = {
        "Example": tf.VarLenFeature(dtype=tf.float32),
        "ExampleIndices": tf.VarLenFeature(dtype=tf.int64),
        "NumExampleVars": tf.FixedLenFeature([1], dtype=tf.int64),
        "Target": tf.VarLenFeature(dtype=tf.float32),
        "TargetIndices": tf.VarLenFeature(dtype=tf.int64),
        "NumTargetVars": tf.FixedLenFeature([1], dtype=tf.int64)
    }

    # Decode multiple examples into a dict of tensors. Each key is a feature name and each value is a tensor with shape
    # batch size.
    parsedBatch = tf.parse_example(batchedSerialisedExamples, featureTypes)

    # Convert sparse tensors to dense tensors. As bag-of-words representations may not all have the same number of
    # values, converting their sparse tensors directly to dense ones necessitates filling in default values. This
    # makes converting from the bag-of-words to a dense vector representation more complicated than if the sparse
    # tensor is used directly.
    if isExamplesBOW:
        parsedBatch["Example"] = sparse_tensor_to_dense.main(
            parsedBatch["Example"], parsedBatch["ExampleIndices"], parsedBatch["NumExampleVars"][0, 0]
        )
    else:
        parsedBatch["Example"] = tf.sparse_tensor_to_dense(parsedBatch["Example"])
    if isTargetsBOW:
        parsedBatch["Target"] = sparse_tensor_to_dense.main(
            parsedBatch["Target"], parsedBatch["TargetIndices"], parsedBatch["NumTargetVars"][0, 0]
        )
    else:
        parsedBatch["Target"] = tf.sparse_tensor_to_dense(parsedBatch["Target"])

    # Split into batches of examples and targets.
    batchExamples = parsedBatch["Example"]
    batchTargets = parsedBatch["Target"]

    return batchExamples, batchTargets
