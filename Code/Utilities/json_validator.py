"""
A cutdown JSON schema implementation validator for Python (with optional unicode to ascii string conversion).

The simplest way to validate a JSON instance against a schema is to call the :func:`main` function.

"""

# Python imports.
import json
import operator
import os
import sys

# Standardise variables and methods needed for both Python versions.
if sys.version_info[0] >= 3:
    basestring = unicode = str
    iteritems = operator.methodcaller("items")
else:
    iteritems = operator.methodcaller("iteritems")


class SchemaError(Exception):
    """Raised when the schema provided is malformed."""

    # TODO
    # TODO add comments and docstring to explain what these things are
    # TODO

    validator = None  # Validator being used when the error occurred.

    def __init__(self, message):
        super(SchemaError, self).__init__(message)
        self.message = message  # Message stating error.
        self.path = []  # Path to the offending item.


class Validator(object):
    """A validator for JSON schema draft version 4."""

    # Define the default types that are allowed.
    defaultTypes = {
        "array": list, "boolean": bool, "integer": int, "null": type(None), "number": (float, int), "object": dict,
        "string": basestring
    }

    def __init__(self, schema, types=None, checkValidity=True):
        """Initialise a validator.

        :param schema:          The JSON instance to validate. This can be either a file location, a JSON object saved
                                as a string or a loaded JSON object.
        :type schema:           str | dict
        :param types:           Any additional or alternative types that should be permitted. This can be used to
                                augment or override the default types used. For example, the default number type is an
                                int or float. To change this to include complex numbers set types to
                                {"number": (float, int, complex)}. If you want only integers to be allowed, then set
                                types to {"number": int}.
        :type types:            dict[str, iterable]
        :param checkValidity:   Whether the schema should be checked for validity against the saved meta-schema.
        :type checkValidity:    bool

        """

        # Load and save the schema.
        if isinstance(schema, basestring):
            if os.path.isfile(schema):
                fid = open(schema, 'r')
                self._schema = json.load(fid)
                fid.close()
            else:
                self._schema = json.loads(schema)
        elif isinstance(schema, dict):
            self._schema = schema
        else:
            raise TypeError("Schema must be a string or dictionary.")

        # Validate the schema against the saved meta-schema if desired.
        if checkValidity:
            self.validate_instance(self._schema, self.metaSchema)

        # Add additional types.
        self._types = dict(self.defaultTypes)
        if types:
            self._types.update(types)
        self._types["any"] = tuple(self._types.values())  # Generate the record for an any type validator.

    def _validate(self, instance, schema=None):
        """Validate an instance of a schema against the schema.

        :param instance:    The JSON object representing an instance of the saved schema.
        :type instance:     dict
        :param schema:      The JSON object of the schema to validate the instance against.
        :type schema:       dict

        """

        # Load the schema associated with this instance if no schema is supplied.
        if not schema:
            schema = self._schema

        # Validate the instance.
        for i, j in iteritems(schema):
            # Determine what validator to use for the given item in the schema.
            validatorName = i.lstrip('$')
            validator = getattr(self, "validate_{:s}".format(validatorName), None)

            # Check whether validation could be performed for the schema item.
            if not validator:
                print("WARNING: No validator found for item {:s} in the schema.".format(validatorName))
                continue

            # Raise all validation errors found.
            errors = validator(j, instance, schema) or None
            for error in errors:
                # Set the validator if it hasn't already been set at a lower level.
                error.validator = error.validator if error.validator else validatorName
                yield error

    def validate_instance(self, instance, schema=None):
        """Validate an instance of a schema against the schema.

        :param instance:    The JSON object representing an instance of the saved schema.
        :type instance:     dict
        :param schema:      The JSON object of the schema to validate the instance against.
        :type schema:       dict

        """

        # Validate the instance.
        for i in self._validate(instance, schema):
            # Raise any errors found.
            raise i

    @classmethod
    def validate_schema(cls, schema):
        """Validate a schema against the saved meta-schema.

        :param schema:  The schema to validate. This can be either a file location, a JSON object saved as
                        a string or a loaded JSON object.
        :type schema:   str | dict

        """

        # Load and schema to validate.
        if isinstance(schema, basestring):
            if os.path.isfile(schema):
                fid = open(schema, 'r')
                schema = json.load(fid)
                fid.close()
            else:
                schema = json.loads(schema)
        elif isinstance(schema, dict):
            pass
        else:
            raise TypeError("Schema must be a string or dictionary.")

        # Validate the schema against the saved meta-schema.
        for i in cls(cls.metaSchema, checkValidity=False)._validate(schema):
            # Raise any errors found.
            error = SchemaError(i.message)
            error.path = i.path
            error.validator = i.validator
            raise error


Validator.metaSchema = {
    "$schema": "http://json-schema.org/draft-03/schema#",
    "id": "http://json-schema.org/draft-03/schema#",
    "type": "object",

    "properties": {
        "type": {
            "type": ["string", "array"],
            "items": {
                "type": ["string", {"$ref": "#"}]
            },
            "uniqueItems": True,
            "default": "any"
        },

        "properties": {
            "type": "object",
            "additionalProperties": {"$ref": "#"},
            "default": {}
        },

        "patternProperties": {
            "type": "object",
            "additionalProperties": {"$ref": "#"},
            "default": {}
        },

        "additionalProperties": {
            "type": [{"$ref": "#"}, "boolean"],
            "default": {}
        },

        "items": {
            "type": [{"$ref": "#"}, "array"],
            "items": {"$ref": "#"},
            "default": {}
        },

        "additionalItems": {
            "type": [{"$ref": "#"}, "boolean"],
            "default": {}
        },

        "required": {
            "type": "boolean",
            "default": False
        },

        "dependencies": {
            "type": "object",
            "additionalProperties": {
                "type": ["string", "array", {"$ref": "#"}],
                "items": {
                    "type": "string"
                }
            },
            "default": {}
        },

        "minimum": {
            "type": "number"
        },

        "maximum": {
            "type": "number"
        },

        "exclusiveMinimum": {
            "type": "boolean",
            "default": False
        },

        "exclusiveMaximum": {
            "type": "boolean",
            "default": False
        },

        "minItems": {
            "type": "integer",
            "minimum": 0,
            "default": 0
        },

        "maxItems": {
            "type": "integer",
            "minimum": 0
        },

        "uniqueItems": {
            "type": "boolean",
            "default": False
        },

        "pattern": {
            "type": "string",
            "format": "regex"
        },

        "minLength": {
            "type": "integer",
            "minimum": 0,
            "default": 0
        },

        "maxLength": {
            "type": "integer"
        },

        "enum": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True
        },

        "default": {
            "type": "any"
        },

        "title": {
            "type": "string"
        },

        "description": {
            "type": "string"
        },

        "format": {
            "type": "string"
        },

        "divisibleBy": {
            "type": "number",
            "minimum": 0,
            "exclusiveMinimum": True,
            "default": 1
        },

        "disallow": {
            "type": ["string", "array"],
            "items": {
                "type": ["string", {"$ref": "#"}]
            },
            "uniqueItems": True
        },

        "extends": {
            "type": [{"$ref": "#"}, "array"],
            "items": {"$ref": "#"},
            "default": {}
        },

        "id": {
            "type": "string",
            "format": "uri"
        },

        "$ref": {
            "type": "string",
            "format": "uri"
        },

        "$schema": {
            "type": "string",
            "format": "uri"
        }
    },

    "dependencies": {
        "exclusiveMinimum": "minimum",
        "exclusiveMaximum": "maximum"
    },

    "default": {}
}


def main(instance, schema):
    """

    :param instance:    The JSON instance to validate. This can be either a file location, a JSON object saved as
                        a string or a loaded JSON object.
    :type instance:     str | dict
    :param schema:      The JSON schema to validate against. This can be either a file location, a JSON object saved as
                        a string or a loaded JSON object.
    :type schema:       str | dict

    """

    validator = Validator(schema)
    validator.validate_instance(instance)
