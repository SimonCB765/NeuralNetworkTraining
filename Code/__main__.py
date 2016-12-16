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

# 3rd party imports.
import jsonschema


# ====================== #
# Create Argument Parser #
# ====================== #
parser = argparse.ArgumentParser(description="Run a specified neural network configuration using Tensorflow.",
                                 epilog="For further information see the README.")

# Mandatory arguments.
parser.add_argument("input", help="The location of the file containing the input examples.")

# Optional arguments.
parser.add_argument("-c", "--config",
                    help="The location of the file containing the configuration parameters to use. "
                         "Default: a file called X_Config.json in the ConfigurationFiles directory where X corresponds"
                         "to the dataType (image, sequence or vector).",
                    type=str)
parser.add_argument("-d", "--dataType",
                    choices=["img", "seq", "vec"],
                    default="vec",
                    help="The type of the data supplied. This is either an image, sequence or single vector per "
                         "example. Default: each example is a single vector.",
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
fileDefaultConfig = os.path.join(
    dirTop, "ConfigurationFiles", "{:s}_Config.json".format(
        "Image" if args.dataType == "img" else ("Sequence" if args.dataType == "seq" else "Vector")
    ))
fileConfigSchema = os.path.join(
    dirTop, "ConfigurationFiles", "{:s}_Schema.json".format(
        "Image" if args.dataType == "img" else ("Sequence" if args.dataType == "seq" else "Vector")
    ))
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
logger = logging.getLogger("DataProcessing")

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
    if args.encode:
        config.set_from_json(fileDefaultConfig, fileConfigSchema, args.encode)
    else:
        config.set_from_json(fileDefaultConfig, fileConfigSchema)
except jsonschema.SchemaError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The configuration schema is not a valid JSON schema. Please correct any changes made to the "
        "schema or download the original and save it at {:s}.\n{:s}".format(fileConfigSchema, str(exceptionInfo[1]))
    )
    isErrors = True
except jsonschema.ValidationError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The default configuration file is not valid against the schema. Please correct any changes made to "
        "the file or download the original and save it at {:s}.\n{:s}".format(fileDefaultConfig, str(exceptionInfo[1]))
    )
    isErrors = True
except jsonschema.RefResolutionError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The configuration schema contains an invalid reference. Please correct any changes made to the "
        "schema or download the original and save it at {:s}.\n{:s}".format(fileConfigSchema, str(exceptionInfo[1]))
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
        try:
            if args.encode:
                config.set_from_json(args.config, fileConfigSchema, args.encode)
            else:
                config.set_from_json(args.config, fileConfigSchema)
        except jsonschema.ValidationError as e:
            exceptionInfo = sys.exc_info()
            logger.error(
                "The user provided configuration file is not valid against the schema.\n{:s}".format(
                    str(exceptionInfo[1]))
            )
            isErrors = True
        except LookupError as e:
            logger.exception("Requested encoding {:s} to convert JSON strings to wasn't found.".format(args.encode))
            isErrors = True

# Display errors if any were found.
if isErrors:
    print("\nErrors were encountered while validating the input arguments. Please see the log file for details.\n")
    sys.exit()

# ================= #
# Shard the Dataset #
# ================= #
if args.dataType == "img":
    # The data is image data.
    if not args.noProcess and config.get_param(["DataProcessing"])[0]:
        logger.info("Now starting the processing of the image data.")
        shard_data.shard_image(fileDataset, dirOutputDataPrep, config, args.target if args.target else None)
elif args.dataType == "seq":
    # The data is sequence data.
    if not args.noProcess and config.get_param(["DataProcessing"])[0]:
        logger.info("Now starting the processing of the sequence data.")
        shard_data.shard_sequence(fileDataset, dirOutputDataPrep, config, args.target if args.target else None)
else:
    # The data is vector data.
    if not args.noProcess and config.get_param(["DataProcessing"])[0]:
        logger.info("Now starting the processing of the vector data.")
        shard_data.shard_vector(fileDataset, dirOutputDataPrep, config, args.target if args.target else None)
