"""Code to shard a large dataset file into multiple small ones."""

# Python imports.
import argparse
import logging
import os
import shutil
import sys

# User imports.
if __package__ != "DataProcessing":
    # The sharding has been executed from the command line not by being imported.
    # Therefore, we need to add the top level Code directory in order to use absolute imports.
    currentDir = os.path.dirname(os.path.join(os.getcwd(), __file__))  # Directory containing this file.
    codeDir = os.path.abspath(os.path.join(currentDir, os.pardir))
    sys.path.append(codeDir)
from Utilities import Configuration

# 3rd party imports.
import jsonschema


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
    dirTop = os.path.abspath(os.path.join(dirCurrent, os.pardir, os.pardir))
    dirOutput = os.path.abspath(os.path.join(dirTop, "ShardedData"))
    dirOutput = args.output if args.output else dirOutput
    fileDefaultConfig = os.path.abspath(os.path.join(dirTop, "ConfigurationFiles", "Config.json"))
    fileConfigSchema = os.path.abspath(os.path.join(dirTop, "ConfigurationFiles", "Schema.json"))
    isErrors = False  # Whether any errors were found.

    # Create the output directory.
    overwrite = args.overwrite
    if overwrite:
        try:
            shutil.rmtree(dirOutput)
        except FileNotFoundError:
            # Can't remove the directory as it doesn't exist.
            pass
        os.makedirs(dirOutput)  # Attempt to make the output directory.
    else:
        try:
            os.makedirs(dirOutput)  # Attempt to make the output directory.
        except FileExistsError as e:
            # Directory already exists so can't continue.
            print("\nCan't continue as the output directory location already exists and overwriting is not enabled.\n")
            sys.exit()

    # Create the logger.
    logger = logging.getLogger("ShardData")
    logger.setLevel(logging.DEBUG)

    # Create the logger file handler.
    fileLog = os.path.join(dirOutput, "Error.log")
    logFileHandler = logging.FileHandler(fileLog)
    logFileHandler.setLevel(logging.DEBUG)

    # Create a console handler for higher level logging.
    logConsoleHandler = logging.StreamHandler()
    logConsoleHandler.setLevel(logging.CRITICAL)

    # Create formatter and add it to the handlers.
    formatter = logging.Formatter("%(name)s\t%(levelname)s\t%(message)s")
    logFileHandler.setFormatter(formatter)
    logConsoleHandler.setFormatter(formatter)

    # Add the handlers to the logger.
    logger.addHandler(logFileHandler)
    logger.addHandler(logConsoleHandler)

    # Validate the input example file.
    fileDataset = args.input
    if not os.path.isfile(fileDataset):
        logger.error("The location containing the input examples does not exist.")
        isErrors = True

    # Validate the file of targets.
    if args.target:
        if not os.path.isfile(args.target):
            logger.error("The supplied location of the file of example targets is not a file.")
            isErrors = True

    # Set default parameter values.
    config = Configuration.Configuration()
    try:
        config.set_from_json(fileDefaultConfig, fileConfigSchema)
    except jsonschema.SchemaError as e:
        logger.exception("The configuration schema is not a valid JSON schema. Please correct any changes made to the "
                         "schema or download the original schema and save it at {:s}".format(fileConfigSchema))
        isErrors = True
    except jsonschema.ValidationError as e:
        logger.exception(
            "The default configuration file is not valid against the schema. Please correct any changes made to the "
            "configuration file or download the original file and save it at {:s}".format(fileDefaultConfig)
        )
        isErrors = True

    # Validate and set any user supplied configuration parameters.
    if args.config:
        if not os.path.isfile(args.config):
            logger.error("The supplied location of the configuration file is not a file.")
            isErrors = True
        else:
            try:
                config.set_from_json(args.config, fileConfigSchema)
            except jsonschema.SchemaError as e:
                logger.exception(
                    "The configuration schema is not a valid JSON schema. Please correct any changes made to the "
                    "schema or download the original schema and save it at {:s}".format(fileConfigSchema)
                )
                isErrors = True
            except jsonschema.ValidationError as e:
                logger.exception("The user provided configuration file is not valid against the schema.")
                isErrors = True

    # Display errors if any were found.
    if isErrors:
        print("\nErrors were encountered while validating the input arguments. Please see the log file for details.\n")
        sys.exit()

    # ================= #
    # Shard the Dataset #
    # ================= #
    if args.target:
        main(fileDataset, dirOutput, config, args.target)
    else:
        main(fileDataset, dirOutput, config)
