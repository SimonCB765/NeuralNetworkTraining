"""This module contains a class, Configuration, that holds the configuration parameters of the running program."""

# Python import.
import json
import operator
import sys

# User imports.
from . import change_json_encoding

# 3rd party imports.
import jsonschema

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    from functools import reduce


class Configuration(object):

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

        # Set schema defaults.
        extrcatedDefaults = self.extract_schema_defaults(schema)
        for i in extrcatedDefaults:
            self.__dict__[i] = extrcatedDefaults[i]

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

    def extract_schema_defaults(self, schema):
        """Extract the default attribute values derived from the types and defaults specified in a schema.

        :param schema:      A JSON object containing a schema that the instantiations will be evaluated against.
        :type schema:       dict
        :return:            The schema defaults.
        :rtype:             dict

        """

        # The returned results for this (sub-)schema. Using two variables enables the returned default (the defaults
        # variable) to be correctly set to None for a null type and stored rather than overlooked as a false value.
        defaults = None
        defaultsFound = False

        # Determine the type of the current (sub-)schema.
        schemaProps = schema.get("properties", {})
        try:
            propsType = schemaProps.get("type", "object")
        except AttributeError:
            propsType = None

        # Extract defaults for the (sub-)schema.
        if propsType == "array":
            # This (sub-)schema is an array. Therefore, we'll set its default value as an empty list if no default
            # is specified.
            defaults = schemaProps.get("default", [])
            defaultsFound = True
        elif propsType == "object":
            # This (sub-)schema is an object. Therefore, we'll set its default value as an empty dictionary if no
            # default is specified and look at the elements it contains.
            defaults = schemaProps.get("default", {})
            defaultsFound = True
            for i in schemaProps:
                # For each element in the current (sub-)schema we need to determine whether the element is an object
                # that contains references or not.
                subschema = schema
                try:
                    # If no exception is raised during the accessing of $ref, then the element is an object that
                    # contains a $ref element. In this case the current (sub-)schema looks something like:
                    #   "CurrentSchema": {
                    #       "description": "Some description...",
                    #       "type": "object",
                    #       "EleWithRef": {"$ref": "#/definitions/DefLocation"},
                    #   }
                    # In this case, rather than going through each element in the EleWithRef object, we directly
                    # replace the {"$ref": "#/definitions/DefLocation"} object with the one located at
                    # #/definitions/DefLocation and go through that instead.
                    defPath = schemaProps[i].get("$ref").split("/")[1:]  # Ignore the '#' at the beginning of the path.
                    subschema["properties"] = reduce(lambda d, key: d.get(key) if d else None, defPath, schema)
                except AttributeError:
                    # The element does not contain a reference, so we just look directly at the element.
                    subschema["properties"] = schemaProps[i]
                subschemaDefaults, defaultExtracted = self.extract_schema_defaults(subschema)
                if defaultExtracted:
                    defaults[i] = subschemaDefaults
        elif propsType in ["boolean", "integer", "number", "string"]:
            # The element is a basic type, so we just need to try and extract a default value.
            defaults = schemaProps.get("default", None)
            if defaults:
                defaultsFound = True
        elif propsType == "null":
            # The element is a null type, so just leave the default as None.
            defaultsFound = True
            defaults = None

        return defaults, defaultsFound
