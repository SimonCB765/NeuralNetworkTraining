"""This module contains a class, Configuration, that holds the configuration parameters of the running program."""

# Python import.
import json

# User imports.
from . import change_json_encoding

# 3rd party imports.
import jsonschema


class Configuration(object):

    # Default variables.
    isLogging = True

    def __init__(self, **kwargs):
        """Initialise a Configuration object.

        :param kwargs:  Keyword arguments to initialise.
        :type kwargs:   dict

        """

        # Initialise any arguments supplied at creation.
        self.set_from_dict(kwargs)

    def set_from_dict(self, paramsToAdd):
        """Set configuration parameters from a dictionary of parameters.

        :param paramsToAdd:  Parameters to add.
        :type paramsToAdd:   dict

        """

        # Initialise any arguments supplied at creation.
        for i in paramsToAdd:
            self.__dict__[i] = paramsToAdd[i]

    def set_from_json(self, config, schema, newEncoding=None):
        """Add parameters to a Configuration object from a JSON formatted file or dict.

        :param config:      The location of a JSON file or a loaded JSON object containing the configuration information
                            to add.
        :type config:       str | dict
        :param schema:      The schema that the configuration information must be validated against. This can either
                            be a file location or a loaded JSON object.
        :type schema:       str | dict
        :param newEncoding: The encoding to convert all strings in the JSON configuration object to.
        :type newEncoding:  str

        """

        # Extract the JSON data.
        if isinstance(config, str):
            fid = open(config, 'r')
            config = json.load(fid)
            if newEncoding:
                change_json_encoding.main(config, newEncoding)
            fid.close()

        # Extract the schema information.
        if isinstance(schema, str):
            fid = open(schema, 'r')
            schema = json.load(fid)
            fid.close()

        # Validate the configuration data.
        jsonschema.validate(config, schema)

        # Add the JSON parameters to the configuration parameters.
        for i in config:
            self.__dict__[i] = config[i]

    def set_from_keyword(self, **kwargs):
        """Set configuration parameters from keywords.

        :param kwargs:  Parameters to add.
        :type kwargs:   dict

        """

        # Initialise any arguments supplied at creation.
        for i in kwargs:
            self.__dict__[i] = kwargs[i]
