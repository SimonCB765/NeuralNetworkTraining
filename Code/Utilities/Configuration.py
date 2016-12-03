"""This module contains a class, Configuration, that holds the configuration parameters of the running program."""

# Python import.
import json

# User imports.
from . import change_json_encoding
from . import extract_JSON_schema_defaults

# 3rd party imports.
import jsonschema


class Configuration(object):

    def __init__(self, **kwargs):
        """Initialise a Configuration object.

        :param kwargs:  Keyword arguments to initialise.
        :type kwargs:   dict

        """

        self._configParams = {}
        self.set_from_dict(kwargs)

    def set_from_dict(self, paramsToAdd):
        """Set configuration parameters from a dictionary of parameters.

        This will overwrite any existing parameters with the same name.

        :param paramsToAdd:  Parameters to add.
        :type paramsToAdd:   dict

        """

        self._configParams.update(paramsToAdd)

    def set_from_json(self, config, schema, newEncoding=None, storeDefaults=1):
        """Add parameters to a Configuration object from a JSON formatted file or dict.

        Any configuration parameters that the user has defined will overwrite existing parameters with the same name.
        Storing defaults will never overwrite user-defined or pre-existing parameters.

        :param config:          The location of a JSON file or a loaded JSON object containing the configuration
                                information to add.
        :type config:           str | dict
        :param schema:          The schema that the configuration information must be validated against. This can either
                                be a file location or a loaded JSON object.
        :type schema:           str | dict
        :param newEncoding:     The encoding to convert all strings in the JSON configuration object to.
        :type newEncoding:      str
        :param storeDefaults:   Whether defaults from the schema should be stored. The possible values are:
                                0 - store no defaults
                                1 - store only defaults that are keys within user defined dictionaries that the user has
                                    not supplied a value for. For example, if the configuration parameter is a
                                    dictionary of three elements and the user has defined one, then allow the other two
                                    to take default values.
                                2+ - store all defaults
        :type storeDefaults:    int

        """

        # Extract the JSON data.
        if isinstance(config, str):
            fid = open(config, 'r')
            config = json.load(fid)
            if newEncoding:
                config = change_json_encoding.main(config, newEncoding)
            fid.close()

        # Extract the schema information.
        if isinstance(schema, str):
            fid = open(schema, 'r')
            schema = json.load(fid)
            if newEncoding:
                schema = change_json_encoding.main(schema, newEncoding)
            fid.close()

        # Validate the configuration data.
        jsonschema.validate(config, schema)

        # Add the configuration parameters.
        self._configParams.update(config)

        # Set schema defaults.
        if storeDefaults:
            extractedDefaults, defaultsExtracted = extract_JSON_schema_defaults.main(schema)
            for i in extractedDefaults:
                # A valid schema will always return a (possibly empty) dictionary following default extraction, so
                # there's no need to check whether this holds any values as the schema has already been validated.
                if i in self._configParams:
                    try:
                        extractedDefaults[i].update(self._configParams[i])
                        self._configParams[i] = extractedDefaults[i]
                    except AttributeError:
                        # We've tried to update a config parameter that is not a dictionary (JSON schema object).
                        # In this case we don't want the default to overwrite the user's defined value.
                        pass
                elif storeDefaults > 1:
                    # The parameter was not defined by the user and we want to store all defaults that will not
                    # overwrite a user's values.
                    self._configParams[i] = extractedDefaults[i]
