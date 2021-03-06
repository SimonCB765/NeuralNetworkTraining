"""Code to run the entire neural network training process."""

# Python imports.
import argparse
import json
import logging.config
import os
import shutil
import sys

# User imports.
from DataProcessing import shard_data
from Libraries.JsonschemaManipulation import Configuration
import Network

# 3rd party imports.
import jsonschema


# ====================== #
# Create Argument Parser #
# ====================== #
parser = argparse.ArgumentParser(description="Run a specified neural network configuration using Tensorflow.",
                                 epilog="For further information see the README.")

# Mandatory arguments.
parser.add_argument("input", help="The location of the file/directory containing the data to use.")

# Optional arguments.
parser.add_argument("-c", "--config",
                    help="The location of the file containing the configuration parameters to use. "
                         "Default: a file called X_Config.json in the ConfigurationFiles directory where X corresponds"
                         "to the dataType (matrix, sequence or vector).",
                    type=str)
parser.add_argument("-d", "--dataType",
                    choices=["mat", "seq", "vec"],
                    default="vec",
                    help="The type of the data supplied. This is either a matrix, sequence or single vector per "
                         "example. Default: vec (each example is a single vector).",
                    type=str)
parser.add_argument("-e", "--encode",
                    default=None,
                    help="The encoding to convert strings in the JSON configuration file to. Default: no "
                         "conversion performed.",
                    type=str)
parser.add_argument("-n", "--noProcess",
                    action="store_true",
                    help="Whether the data should be prevented from being processed. Default: data can be processed.")
parser.add_argument("-o", "--output",
                    help="The location of the directory to save the output to. Default: a top level "
                         "directory called Output.",
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
dirTop = os.path.abspath(os.path.join(dirCurrent, os.pardir))
dirOutput = os.path.join(dirTop, "Output")
dirOutput = args.output if args.output else dirOutput
dirOutputDataPrep = os.path.join(dirOutput, "DataProcessing")

# Create the output directory.
overwrite = args.overwrite
if overwrite:
    try:
        shutil.rmtree(dirOutput)
    except FileNotFoundError:
        # Can't remove the directory as it doesn't exist.
        pass
    os.makedirs(dirOutput)  # Attempt to make the output directory.
    os.makedirs(dirOutputDataPrep)  # Attempt to make the data preparation output directory.
else:
    try:
        os.makedirs(dirOutput)  # Attempt to make the output directory.
        os.makedirs(dirOutputDataPrep)  # Attempt to make the data preparation output directory.
    except FileExistsError as e:
        # Directory already exists so can't continue.
        print("\nCan't continue as the output directory location already exists and overwriting is not enabled.\n")
        sys.exit()

# Create the logger. In order to do this we need to overwrite the value in the configuration information that records
# the location of the file that the logs are written to.
fileLoggerConfig = os.path.join(dirTop, "ConfigurationFiles", "Loggers.json")
fileLogOutput = os.path.join(dirOutput, "Logs.log")
logConfigInfo = json.load(open(fileLoggerConfig, 'r'))
logConfigInfo["handlers"]["file"]["filename"] = fileLogOutput
logging.config.dictConfig(logConfigInfo)
logger = logging.getLogger("__main__")

# Determine the location of the schema and default configuration files.
dirConfiguration = os.path.join(dirTop, "ConfigurationFiles")
fileDefaultConfig = os.path.join(  # The default configuration file for the given input type.
    dirConfiguration, "{:s}_Config.json".format(
        "Matrix" if args.dataType == "mat" else ("Sequence" if args.dataType == "seq" else "Vector")
    ))
fileBaseSchema = os.path.join(dirConfiguration, "Base_Schema.json")
fileTypedSchema = os.path.join(  # The schema for the specific input type.
    dirConfiguration, "{:s}_Schema.json".format(
        "Matrix" if args.dataType == "mat" else ("Sequence" if args.dataType == "seq" else "Vector")
    ))
isErrors = False  # Whether any errors were found.

# Create the configuration object.
config = Configuration.Configuration()

# Validate the base schema and set default values from it.
try:
    if args.encode:
        config.set_from_json(fileDefaultConfig, fileBaseSchema, args.encode, schemaRoot=dirConfiguration)
    else:
        config.set_from_json(fileDefaultConfig, fileBaseSchema, schemaRoot=dirConfiguration)
except jsonschema.SchemaError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The base schema is not a valid JSON schema. Please correct any changes made to the "
        "schema or download the original and save it at {:s}.\n{:s}".format(fileBaseSchema, str(exceptionInfo[1]))
    )
    isErrors = True
except jsonschema.ValidationError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The default configuration file is not valid against the base schema. Please correct any changes made to "
        "the file or download the original and save it at {:s}.\n{:s}".format(fileDefaultConfig, str(exceptionInfo[1]))
    )
    isErrors = True
except jsonschema.RefResolutionError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The base schema contains an invalid reference. Please correct any changes made to the "
        "schema or download the original and save it at {:s}.\n{:s}".format(fileBaseSchema, str(exceptionInfo[1]))
    )
    isErrors = True
except LookupError as e:
    logger.exception("Requested encoding {:s} to convert JSON strings to wasn't found.".format(args.encode))
    isErrors = True

# Validate the input type specific schema and set default values from it.
try:
    if args.encode:
        config.set_from_json(fileDefaultConfig, fileTypedSchema, args.encode, schemaRoot=dirConfiguration)
    else:
        config.set_from_json(fileDefaultConfig, fileTypedSchema, schemaRoot=dirConfiguration)
except jsonschema.SchemaError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The typed schema is not a valid JSON schema. Please correct any changes made to the "
        "schema or download the original and save it at {:s}.\n{:s}".format(fileTypedSchema, str(exceptionInfo[1]))
    )
    isErrors = True
except jsonschema.ValidationError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The default configuration file is not valid against the typed schema. Please correct any changes made to "
        "the file or download the original and save it at {:s}.\n{:s}".format(fileDefaultConfig, str(exceptionInfo[1]))
    )
    isErrors = True
except jsonschema.RefResolutionError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The typed schema contains an invalid reference. Please correct any changes made to the "
        "schema or download the original and save it at {:s}.\n{:s}".format(fileTypedSchema, str(exceptionInfo[1]))
    )
    isErrors = True
except LookupError as e:
    logger.exception("Requested encoding {:s} to convert JSON strings to wasn't found.".format(args.encode))
    isErrors = True

# Validate and set any user supplied configuration parameters.
if args.config:
    if not os.path.isfile(args.config):
        logger.error("The supplied location of the configuration file is not a file.")
        isErrors = True
    else:
        # Set user configuration parameters validated against the base schema.
        try:
            if args.encode:
                config.set_from_json(args.config, fileBaseSchema, args.encode)
            else:
                config.set_from_json(args.config, fileBaseSchema)
        except jsonschema.ValidationError as e:
            exceptionInfo = sys.exc_info()
            logger.error(
                "The user provided configuration file is not valid against the base schema.\n{:s}".format(
                    str(exceptionInfo[1]))
            )
            isErrors = True
        except LookupError as e:
            logger.exception("Requested encoding {:s} to convert JSON strings to wasn't found.".format(args.encode))
            isErrors = True

        # Set user configuration parameters validated against the input specific schema.
        try:
            if args.encode:
                config.set_from_json(args.config, fileTypedSchema, args.encode)
            else:
                config.set_from_json(args.config, fileTypedSchema)
        except jsonschema.ValidationError as e:
            exceptionInfo = sys.exc_info()
            logger.error(
                "The user provided configuration file is not valid against the typed schema.\n{:s}".format(
                    str(exceptionInfo[1]))
            )
            isErrors = True
        except LookupError as e:
            logger.exception("Requested encoding {:s} to convert JSON strings to wasn't found.".format(args.encode))
            isErrors = True

# Display any errors found while validating the configuration file.
if isErrors:
    print("\nErrors were encountered while validating the configuration file. Please see the log file for details.\n")
    sys.exit()

# Determine whether data processing is to take place.
isProcessing = False
if not args.noProcess and config.get_param(["DataProcessing"])[0]:
    isProcessing = True

# Validate the input data.
inputData = args.input
if not os.path.exists(inputData):
    logger.error("The location containing the input data does not exist.")
    isErrors = True
if isProcessing and (not os.path.isfile(inputData)):
    logger.error("The input dataset to be processed is not a file.")
    isErrors = True
elif (not isProcessing) and (not os.path.isdir(inputData)):
    logger.error("No processing is selected. The input data should therefore be a directory, but isn't.")
    isErrors = True

# Validate the file of targets.
if args.target:
    if not os.path.isfile(args.target):
        logger.error("The supplied location of the file of example targets is not a file.")
        isErrors = True

# Display errors if any were found.
if isErrors:
    print("\nErrors were encountered while validating the input arguments. Please see the log file for details.\n")
    sys.exit()

# ================= #
# Shard the Dataset #
# ================= #
if args.dataType == "mat":
    # The data is image data.
    if isProcessing:
        logger.info("Now starting the processing of the matrix data.")
        shard_data.shard_matrix(inputData, dirOutputDataPrep, config, args.target if args.target else None)
elif args.dataType == "seq":
    # The data is sequence data.
    if isProcessing:
        logger.info("Now starting the processing of the sequence data.")
        shard_data.shard_sequence(inputData, dirOutputDataPrep, config, args.target if args.target else None)
else:
    # The data is vector data.

    # Can only use bag-of-words data when there is a header present.
    isExamplesBOW = config.get_param(["ExampleBOW"])[1]
    if isExamplesBOW:
        logger.info("The example data is bag-of-words, so the data is treated as having a header.")
        config.set_param(["DataProcessing", "Examples", "HeaderPresent"], True, overwrite=True)
    isTargetsBOW = config.get_param(["TargetBOW"])[1]
    if isTargetsBOW:
        logger.info("The target data is bag-of-words, so the data is treated as having a header.")
        config.set_param(["DataProcessing", "Targets", "HeaderPresent"], True, overwrite=True)

    # Perform the processing and sharding if needed.
    if isProcessing:
        logger.info("Now starting the processing of the vector data.")
        shard_data.shard_vector(inputData, dirOutputDataPrep, config, args.target if args.target else None)

    # Perform the training/testing/validation of the network.
    Network.main.main_vector(dirOutput if isProcessing else inputData, config)
