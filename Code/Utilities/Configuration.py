"""This module contains a class, Configuration, that holds the configuration parameters of the running program."""

# Python import.
import json
import sys

# User imports.
from . import json_to_ascii


class Configuration(object):

    # Default variables.
    isLogging = True

    def __init__(self, *args, **kwargs):
        """Initialise a Configuration object.

        Parameters supplied as kwargs take precedence. Therefore, any kwargs supplied that are also in one of the args
        files will overwrite the args file value. Similarly, args files later in the list of arguments will take
        precedence over earlier ones.

        :param args:        The location of files containing JSON formatted configuration information that should
                            be used to initialise the Configuration object.
        :type args:         tuple
        :param kwargs:      Any additional configuration parameters to set.
        :type kwargs:       dict

        """

        # Initialise the configuration parameters from JSON files.
        for i in args:
            self.set_from_json(i)

        # Initialise any arguments supplied at creation.
        for i in kwargs:
            self.__dict__[i] = kwargs[i]

    def set_from_json(self, fileConfig):
        """Add parameters to a Configuration object from a JSON formatted file.

        :param fileConfig:  The location of a file containing JSON formatted configuration information.
        :type fileConfig:   str

        """

        # Extract the JSON data.
        fid = open(fileConfig, 'r')
        configData = json.load(fid)
        fid.close()

        # Convert the JSON data to ascii if needed.
        versionNum = sys.version_info[0]  # Determine major version number.
        if versionNum == 2:
            configData = json_to_ascii.main(configData)

        # Add the JSON parameters to the configuration parameters.
        for i in configData:
            self.__dict__[i] = configData[i]
