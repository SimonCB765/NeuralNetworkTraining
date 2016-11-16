"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import argparse
import os
import shutil
import sys

# User imports.
from Code.TensorFlow.DataProcessing import Configuration


def main(fileExamples, dirOutput, fileTargets=None):
    pass


def parse_config():
    pass


if __name__ == "__main__":
    # ====================== #
    # Create Argument Parser #
    # ====================== #
    parser = argparse.ArgumentParser(description="Shard a large dataset file into multiple smaller ones.",
                                     epilog="The dataset is assumed to contain one vector per datapoint. For further "
                                            "information see the README.")

    # Mandatory arguments.
    parser.add_argument("examples", help="The location of the file containing the input examples.")
    parser.add_argument("output", help="The location of the directory to write the output to.")

    # Optional arguments.
    parser.add_argument("-c", "--config",
                        help="The location of the file containing the configuration parameters to use. "
                             "Default: a file called Sequence.json in the ConfigurationFiles/DataProcessing directory.",
                        type=str)
    parser.add_argument("-t", "--target",
                        help="The location of the file containing the target values/vectors for each input example. "
                             "Default: target data is not used.",
                        type=str)
    parser.add_argument("-w", "--overwrite",
                        action="store_true",
                        help="Whether the output directory should be overwritten. Default: do not overwrite.")

# ============================ #
# Parse and Validate Arguments #
# ============================ #
args = parser.parse_args()
dirCurrent = os.path.dirname(os.path.join(os.getcwd(), __file__))  # Directory containing this file.
dirTop = os.path.abspath(os.path.join(dirCurrent, os.pardir, os.pardir, os.pardir))
dirData = os.path.abspath(os.path.join(dirTop, "Data"))
errorsFound = []  # Container for any error messages generated during the validation.

# Set default parameter values.
config = Configuration.Configuration()

# Validate the input dataset.
fileDataset = args.input
if not os.path.isfile(fileDataset):
    errorsFound.append("The input dataset location does not contain a file.")

# Validate the number of datapoints per file.
if args.number < 1:
    errorsFound.append("The number of datapoints per file must be at least one.")

# Validate the output directory.
dirOutput = os.path.abspath(os.path.join(dirData, "ShardedData"))
dirOutput = args.output if args.output else dirOutput
overwrite = args.overwrite
if overwrite:
    try:
        shutil.rmtree(dirOutput)
    except FileNotFoundError:
        # Can't remove the directory as it doesn't exist.
        pass
    except Exception as e:
        # Can't remove the directory for another reason.
        errorsFound.append("Could not overwrite the output directory location - {0:s}".format(str(e)))
elif os.path.exists(dirOutput):
    errorsFound.append("The output directory location already exists and overwriting is not enabled.")

# Validate the column ignore file.
fileIgnoreColumns = os.path.join(dirData, "IgnoredColumns.txt")
fileIgnoreColumns = args.ignore if args.ignore else fileIgnoreColumns
if not os.path.exists(fileIgnoreColumns):
    errorsFound.append("The location of the file containing the columns to ignore does not exist.")
elif not os.path.isfile(fileIgnoreColumns):
    errorsFound.append("The location of the file containing the columns to ignore is not a file.")

# Display errors if any were found.
if errorsFound:
    print("\n\nThe following errors were encountered while parsing the input arguments:\n")
    print('\n'.join(errorsFound))
    sys.exit()

# Only create the output directory if there were no errors encountered.
try:
    os.makedirs(dirOutput, exist_ok=True)  # Attempt to make the output directory. Don't care if it already exists.
except Exception as e:
    print("\n\nThe following errors were encountered while parsing the input arguments:\n")
    print("The output directory could not be created - {0:s}".format(str(e)))
    sys.exit()

# ====================== #
# Shard the Dataset File #
# ====================== #
