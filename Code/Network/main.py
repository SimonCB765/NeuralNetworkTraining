"""Code to coordinate the running of the network training/testing/validation."""

# Python imports.
import json
import logging
import os

# User imports.
from . import InputPipeline
from . import Structure

# 3rd party imports.
import tensorflow as tf

# Globals.
LOGGER = logging.getLogger(__name__)


def main_vector(dirData, config):
    """Run network operations for vector data.

    :param dirData:     The location of the directory containing the data to use.
    :type dirData:      str
    :param config:      The object containing the configuration parameters for the network.
    :type config:       JsonschemaManipulation.Configuration

    """

    # Define the location where the processed data is recorded.
    dirProcessedData = os.path.join(dirData, "DataProcessing")

    # Determine the number of example and target variables.
    fileNumVars = os.path.join(dirProcessedData, "NumVariables.json")
    fidNumVars = open(fileNumVars, 'r')
    numberVariables = json.load(fidNumVars)
    fidNumVars.close()
    numExampleVars = numberVariables["NumExampleVariables"]
    numTargetVars = numberVariables["NumTargetVariables"]

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Set the graph-level random seed.
        randomSeed = config.get_param(["RandomSeed"])[1]
        tf.set_random_seed(randomSeed)

        # Setup the input pipeline that generates mini-batches.
        LOGGER.info("Now setting up the input pipeline.")
        batchExamples, batchTargets = InputPipeline.vector.main(dirProcessedData, config)

        # Setup the network structure. The network is built in a four stage approach:
        #   1) inference()  - This operation will build the graph as far as is needed to make predictions (i.e. up to
        #                     the end of a forward pass).
        #   2) loss()       - This will add to the graph the operations required to calculate the loss.
        #   3) training()   - This will add to the graph the operations required to compute and apply gradients.
        #   4) evaluation() - This will add to the graph the operations required to perform an evaluation of the
        #                     performance of the network.
        # A network construction example can be found https://www.tensorflow.org/get_started/mnist/mechanics and code
        # https://www.tensorflow.org/get_started/mnist/mechanics.
        LOGGER.info("Now creating the network.")
        networkType = config.get_param(["Network", "NetworkType"])[1]
        if networkType == "autoencoder":
            # Create an autoencoder network.
            inference = Structure.denoising_autoencoder.inference(batchExamples, numExampleVars, config)
        LOGGER.info("Network created.")

        # Create the operation that will create the graph, etc.
        try:
            initOp = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        except AttributeError:
            initOp = tf.group(tf.initialize_all_variables(), tf.local_variables_initializer())

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
                inf = session.run(inference)
        except tf.errors.OutOfRangeError:
            LOGGER.info("Done training. Epoch limit reached.")
        finally:
            # When done, ask the threads to stop.
            coordinator.request_stop()

        # Wait for the threads to finish.
        coordinator.join(threads)
        session.close()
