"""Code to coordinate the running of the network training/testing/validation."""

# Python imports.
import logging
import os

# User imports.
from . import InputPipeline

# 3rd party imports.
import tensorflow as tf

# Globals.
LOGGER = logging.getLogger(__name__)


def main_vector(dirData, config):
    """Run network operations for vector data.

    :param dirData:     The location of the directory containing the data to use.
    :type dirData:      str
    :param config:      The object containing the configuration parameters for the sharding.
    :type config:       JsonschemaManipulation.Configuration

    """

    # Setup the input pipeline that generates mini-batches.
    dirShardedFiles = os.path.join(dirData, "DataProcessing")
    batchExamples, batchTargets = InputPipeline.vector.main(dirShardedFiles, config)

    # Create the operation that will create the graph, etc.
    try:
        initOp = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    except AttributeError:
        initOp = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    # Create a session for running operations in the Graph.
    session = tf.Session()

    # Initialize the variables (like the epoch counter).
    session.run(initOp)

    # Start input enqueue threads.
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

    # Run the created training operations.
    try:
        # Run the training operations until the coordinator says to stop.
        while not coordinator.should_stop():
            examples, targets = session.run([batchExamples, batchTargets])
    except tf.errors.OutOfRangeError:
        LOGGER.info("Done training. Epoch limit reached.")
    finally:
        # When done, ask the threads to stop.
        coordinator.request_stop()

    # Wait for the threads to finish.
    coordinator.join(threads)
    session.close()
