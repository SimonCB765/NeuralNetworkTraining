"""This module contains a class, Configuration, that holds the configuration parameters of the running program."""

# Python import.
import json

# User imports.
from . import json_schema_operations

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

    def get_param(self, path):
        """Extract a configuration parameter from the dictionary of parameters.

        :param path:    The path through the configuration parameter dictionary to use to extract the parameter.
        :type path:     list[str]
        :return:        The parameter's value or an indication that the parameter does not exist.
        :rtype:         bool, list | int | str | float | dict | None
                            1st element indicates whether the parameter was found.
                            2nd element is the parameter's value if found and the name of the first missing dictionary
                                key if not found (e.g. "A" if self._configParams["B"]["A"]["C"] fails because "A"
                                is not a key in the self._configParams["B"] dictionary.

        """

        paramFound = False  # Whether the parameter was found.
        paramValue = self._configParams  # The value to return.
        for i in path:
            if i in paramValue:
                # The next element in the path was found, so keep looking.
                paramValue = paramValue[i]
            else:
                # The parameter was not found, so terminate the search.
                paramValue = i
                break
        else:
            # The parameter was found as all elements in the path were found.
            paramFound = True

        return paramFound, paramValue

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
                                0 - Store no defaults.
                                1 - Store top level parameter defaults (i.e. those contained directly within the
                                    properties element of the schema). Also store defaults for missing values within
                                    sub-schemas that a user has defined some values of. For examples, if the
                                    configuration parameter is a dictionary of three elements and the user has given a
                                    value for one, then allow the other two to take default values. If the user had not
                                    given values for any of the three, then no defaults would be set.
                                2+ - Store all defaults.
        :type storeDefaults:    int

        """

        # Extract the JSON data.
        if isinstance(config, str):
            fid = open(config, 'r')
            config = json.load(fid)
            if newEncoding:
                config = json_schema_operations.change_encoding(config, newEncoding)
            fid.close()

        # Extract the schema information.
        if isinstance(schema, str):
            fid = open(schema, 'r')
            schema = json.load(fid)
            if newEncoding:
                schema = json_schema_operations.change_encoding(schema, newEncoding)
            fid.close()

        # Validate the configuration data.
        jsonschema.validate(config, schema)

        # Add the configuration parameters.
        self._configParams.update(config)

        # Set schema defaults.
        if storeDefaults:
            extractedDefaults, defaultsExtracted = json_schema_operations.extract_schema_defaults(schema)
            for i in extractedDefaults:
                # A valid schema will always return a (possibly empty) dictionary following default extraction, so
                # there's no need to check whether this holds any values as the schema has already been validated.
                if i in self._configParams:
                    # The user has specified some (or all) of the values for this configuration parameter.
                    try:
                        extractedDefaults[i].update(self._configParams[i])
                        self._configParams[i] = extractedDefaults[i]
                    except AttributeError:
                        # We've tried to update a config parameter that is not a dictionary (JSON schema object).
                        # In this case we don't want the default to overwrite the user's defined value.
                        pass
                elif not isinstance(i, dict):
                    # The default is a top level parameter (held directly in the schema's properties element) that
                    # the user did not supply a value for, so we use the default.
                    self._configParams[i] = extractedDefaults[i]
                elif storeDefaults > 1:
                    # The parameter was not defined by the user and we want to store all defaults that will not
                    # overwrite a user's values.
                    self._configParams[i] = extractedDefaults[i]
