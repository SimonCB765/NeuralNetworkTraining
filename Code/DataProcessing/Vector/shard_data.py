"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import argparse
import os
import shutil
import sys

# User imports.
if __package__ != "Vector":
    # The sharding has been executed from the command line not from being imported into another module.
    # Therefore, we need to add the top level Code directory in order to use absolute imports.
    currentDir = os.path.dirname(os.path.join(os.getcwd(), __file__))  # Directory containing this file.
    codeDir = os.path.abspath(os.path.join(currentDir, os.pardir, os.pardir))
    sys.path.append(codeDir)
from Utilities import Configuration


def main(fileExamples, dirOutput, config, fileTargets=None):
    """Shard the dataset of examples.

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


if __name__ == "__main__":
    # ====================== #
    # Create Argument Parser #
    # ====================== #
    parser = argparse.ArgumentParser(description="Shard a large dataset file into multiple smaller ones.",
                                     epilog="The dataset is assumed to contain one vector per datapoint. For further "
                                            "information see the README.")

    # Mandatory arguments.
    parser.add_argument("input", help="The location of the file containing the input examples.")

    # Optional arguments.
    parser.add_argument("-c", "--config",
                        help="The location of the file containing the configuration parameters to use. "
                             "Default: a file called Sequence.json in the ConfigurationFiles/DataProcessing directory.",
                        type=str)
    parser.add_argument("-o", "--output",
                        help="The location of the directory to save the sharded output to. Default: a top level "
                             "directory called ShardedData.",
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
    fileDefaultConfig = os.path.abspath(os.path.join(dirTop, "ConfigurationFiles", "DataProcessing", "Vector.json"))
    fileConfigSchema = os.path.abspath(os.path.join(dirTop, "ConfigurationFiles", "DataProcessing", "Schema.json"))
    errorsFound = []  # Container for any error messages generated during the validation.

    # Set default parameter values.
    config = Configuration.Configuration()
    config.set_from_json(fileDefaultConfig, fileConfigSchema)

    # Validate the input example file.
    fileDataset = args.input
    if not os.path.isfile(fileDataset):
        errorsFound.append("The location containing the input examples does not exist.")

    # Validate the file of targets.
    if args.target:
        if not os.path.isfile(args.target):
            errorsFound.append("The supplied location of the file of example targets is not a file.")

    # Validate the output directory.
    dirOutput = os.path.abspath(os.path.join(dirTop, "ShardedData"))
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

    # Display errors if any were found.
    if errorsFound:
        print("\n\nThe following errors were encountered while parsing the input arguments:\n")
        print('\n'.join(errorsFound))
        sys.exit()

    # Validate and set any user supplied configuration parameters.
    if args.config:
        if not os.path.isfile(args.config):
            print("\n\nThe following errors were encountered while parsing the input arguments:\n")
            print("The supplied location of the configuration file is not a file.")
            sys.exit()
        else:
            config.set_from_json(args.config, fileConfigSchema)

    # Only create the output directory if there were no errors encountered.
    try:
        os.makedirs(dirOutput, exist_ok=True)  # Attempt to make the output directory. Don't care if it already exists.
    except Exception as e:
        print("\n\nThe following errors were encountered while parsing the input arguments:\n")
        print("The output directory could not be created - {0:s}".format(str(e)))
        sys.exit()

    # ================= #
    # Shard the Dataset #
    # ================= #
    if args.target:
        main(fileDataset, dirOutput, config, args.target)
    else:
        main(fileDataset, dirOutput, config)
