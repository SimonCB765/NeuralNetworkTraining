"""Code to vectorise a bag-of-words."""

# 3rd party imports.
import tensorflow as tf


def main(values, indices, numVars, fillValue=-1.):
    """Convert a bag-of-words representation of a tensor to a dense tensor.

    The general way that this can be done by replacing the column indices and shape in the values tensor, and then
    converting the values tensor to a dense tensor. Assume we have two datapoints and ten variables. Our datapoints are:
        Datapoint_1 has values for indices/columns 1, 3, 4 and 6 (with no values for 0, 2, 5, 7, 8 and 9)
        Datapoint_2 has values for indices/columns 0, 1, 3, 5, 6 and 8 (with no values for 2, 4, 7 and 9)
    Our values tensor will then look something like:
        SparseTensor(
                        indices=[
                                    [0, 1], [0, 2], [0, 3], [0, 5]
                                    [1, 0], [1, 1], [1, 2], [1, 4], [1, 5], [1, 6]
                                ],
                        values=...,
                        shape=[2, 7]
                    )
    We change the shape of the tensor from [2, 7] to [2, 10] as we know there are ten variables/columns/indices and we
    alter the indices in order to get:
        SparseTensor(
                        indices=[
                                    [0, 1], [0, 3], [0, 4], [0, 6]
                                    [1, 0], [1, 1], [1, 3], [1, 5], [1, 6], [1, 8]
                                ],
                        values=...,
                        shape=[2, 10]
                    )

    :param values:      The values to place in the dense tensor.
    :type values:       tf.SparseTensorValue
    :param indices:     The indices where the values should be placed in the dense tensor.
    :type indices:      tf.SparseTensorValue
    :param numVars:     The number of variables needed to fill out the dense tensor.
    :type numVars:      int
    :param fillValue:   The value to fill in the empty indices with.
    :type fillValue:    float
    :return:            A dense tensor created from the bag-of-words tensor.
    :rtype:             tf.Tensor

    """

    # Determine the shape of the output tensor.
    shape = [values.shape[0], numVars]

    # Determine the indices that values are provided for. Do this by substituting the correct column indices in for the
    # ones that are recorded when decoding examples. Assume we have two datapoints and
    # For example, if we have two datapoints and 10 variables, then a
    # the datapoint should have values for columns
    # 1, 3, 4 and 6 (with columns 0, 2, 5, 7 and 8 having no values), then the decoded sparse tensor will be something
    # like:
    #   SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3]])
    rows = tf.unpack(values.indices, axis=1)[0]  # Take the record of the rows from the values
    cols = indices.values  # and the record of the columns from the values of the indices sparse tensor
    newIndices = tf.pack([rows, cols], axis=1)  # and combine them to get a single (2, X) tensor.

    # Create the new tensor, first as a sparse tensor and then as a dense one.
    output = tf.SparseTensor(indices=newIndices, values=values.values, shape=shape)
    output = tf.sparse_tensor_to_dense(output, default_value=fillValue)

    return output
