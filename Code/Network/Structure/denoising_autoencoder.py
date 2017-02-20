"""Functions to implement vanilla and denoising autoencoders."""

# Python imports.
import logging
import math

# 3rd party imports.
import tensorflow as tf

# Globals.
LOGGER = logging.getLogger(__name__)


def evaluation():
    pass


def inference(examples, numExampleVars, config):
    """Build the network graph up to the point where it can be used to make predictions (end of the forward pass).

    :param examples:        A tensor containing a batch of processed examples.
    :type examples:         Dense tensor of type tf.float32
    :param numExampleVars:  The number of variables in each example.
    :type numExampleVars:   int
    :param config:          The object containing the configuration parameters for the network.
    :type config:           JsonschemaManipulation.Configuration
    :return:
    :rtype:

    """

    # Extract the information about the layers of the network from the configuration file.
    networkLayers = config.get_param(["Network", "Layers"])[1]

    # Create the non-input layers.
    layerOutputs = {}
    for i, j in enumerate(networkLayers):
        # Determine the number of nodes in the layer and the activation function to use.
        numNodesInLayer = j["NumberNodes"]
        activationFunc = j["ActivationFunc"]

        # Create the weights and biases for the layer.
        with tf.name_scope("{:s}".format("output" if i == (len(networkLayers) - 1) else "hidden{:d}".format(i))):
            # Log the creation of the layer.
            LOGGER.info("Creating layer {:d} with {:d} nodes.".format(i, numNodesInLayer))

            # Determine number of nodes and activations from the previous layer. If the current layer is the first
            # defined layer, then the previous layer is the input examples.
            numSourceNodes = numExampleVars if i == 0 else networkLayers[i - 1]["NumberNodes"]
            prevLayerOutput = examples if i == 0 else layerOutputs[i - 1]

            # Create the weights, biases and outputs for the layer.
            weights = tf.Variable(
                tf.truncated_normal([numSourceNodes, numNodesInLayer], stddev=1.0 / math.sqrt(float(numNodesInLayer))),
                name="weights"
            )
            biases = tf.Variable(tf.zeros([numNodesInLayer]), name="biases")
            if activationFunc == "relu":
                outputs = tf.nn.relu(tf.matmul(prevLayerOutput, weights) + biases)
            elif activationFunc == "sigmoid":
                outputs = tf.nn.sigmoid(tf.matmul(prevLayerOutput, weights) + biases)
            else:
                # Linear activation.
                outputs = tf.matmul(prevLayerOutput, weights) + biases
            layerOutputs[i] = outputs

    return layerOutputs[len(networkLayers) - 1]


def loss():
    pass


def training():
    pass
