"""
A cutdown JSON schema implementation validator for Python (with optional unicode to ascii string conversion).

As per the JSON schema, keywords only limit the range of values of certain primitive types. For example,
"the 'maxLength' keyword will only restrict certain strings (that are too long) from being valid. If the instance is a
number, boolean, null, array, or object, the keyword passes validation". Additionally, validation keywords that are
missing do not prevent validation occurring. The validation therefore permits everything except when validation
explicitly fails.

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
    """Exception raised when the schema provided is malformed.

    The relevant attributes of the exception are the same as those of the ValidationError.

    """

    validator = None  # Validator being used when the exception was raised.

    def __init__(self, message):
        """Initialise a schema error exception.

        :param message: The message associated with the exception.
        :type message:  str

        """

        super(SchemaError, self).__init__(message)
        self.message = message
        self.path = []


class UnknownType(Exception):
    """Exception to indicate that an unknown type was supplied in a schema."""


class ValidationError(Exception):
    """Exception to indicate that a property is not valid.

    The relevant attributes for the exception are:
        message - A message explaining the exception.
        path - A list containing the path to reach the invalid property.
        validator - The validator being used when the exception was raised.

    """

    validator = None   # Validator being used when the exception was raised.

    def __init__(self, message):
        """Initialise a validation error exception.

        :param message: The message associated with the exception.
        :type message:  str

        """

        super(ValidationError, self).__init__(message)
        self.message = message
        self.path = []


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
        :type types:            dict[str, tuple]
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
            types = {i: tuple(j) for i, j in iteritems(types)}
            self._types.update(types)
        self._types["any"] = tuple(self._types.values())  # Generate the record for an any type validator.

    def _is_type(self, instance, typeToCheck):
        """Determine if a given instance is of the specified type.

        :param instance:    The object that is to have its type checked.
        :type instance:     object
        :param typeToCheck: The type to check the instance's type against.
        :type typeToCheck:  object
        :return:            Whether the instance is of the specified type.
        :rtype:             bool

        """

        # Establish whether the type is a permissible one according to the preset types.
        if typeToCheck not in self._types:
            raise UnknownType(typeToCheck)

        # Check whether the instance is of the given type.
        types = self._types[typeToCheck]
        if isinstance(instance, bool):
            # As bool inherits from int, this needs to be accounted for separately.
            if int in types and bool not in types:
                raise False
        return isinstance(instance, types)

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
            validator = getattr(self, "_validate_{:s}".format(validatorName), None)

            # Check whether validation could be performed for the schema item.
            if not validator:
                print("WARNING: No validator found for item {:s} in the schema.".format(validatorName))
                continue

            # Raise all validation errors found.
            errors = validator(j, instance, schema)
            errors = errors if errors else ()  # If no errors were found, then set errors to an empty iterable.
            for error in errors:
                # Set the validator if it hasn't already been set at a lower level.
                error.validator = error.validator if error.validator else validatorName
                yield error

    def _validate_maximum(self, maximum, instance, schema):
        """Validate a maximum constraint holds for the instance.

        :param maximum:     The maximum to validate against.
        :type maximum:      int | float
        :param instance:    The schema instance being validated.
        :type instance:     int | float
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate that the instance's value meets the maximum constraint.
        if self._is_type(instance, "number"):
            # Only perform validation if the instance is a number.
            exclsiveMax = schema.get("exclusiveMaximum", False)
            if self._is_type(exclsiveMax, "boolean"):
                if exclsiveMax:
                    # The maximum is exclusive.
                    isValid = instance < maximum
                else:
                    # The maximum is not exclusive.
                    isValid = instance <= maximum

                if not isValid:
                    yield ValidationError("{:s} is greater than{:s} the maximum value of {:s}.".format(
                        str(instance), (" or equal to" if exclsiveMax else ""), str(maximum)
                    ))

    def _validate_maxLength(self, maxLength, instance, schema):
        """Validate a maximum length constraint holds for the instance.

        :param maxLength:   The maximum length to validate against.
        :type maxLength:    int | float
        :param instance:    The schema instance being validated.
        :type instance:     str
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        if self._is_type(maxLength, "integer") and maxLength >= 0 and self._is_type(instance, "string") and \
                len(instance) > maxLength:
            yield ValidationError("{:s} is longer than the maximum length of {:d}.".format(instance, maxLength))

    def _validate_minimum(self, minimum, instance, schema):
        """Validate a minimum constraint holds for the instance.

        :param minimum:     The minimum to validate against.
        :type minimum:      int | float
        :param instance:    The schema instance being validated.
        :type instance:     int | float
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate that the instance's value meets the minimum constraint.
        if self._is_type(instance, "number"):
            # Only perform validation if the instance is a number.
            exclsiveMin = schema.get("exclusiveMinimum", False)
            if self._is_type(exclsiveMin, "boolean"):
                if exclsiveMin:
                    # The minimum is exclusive.
                    isValid = instance > minimum
                else:
                    # The minimum is not exclusive.
                    isValid = instance >= minimum

                if not isValid:
                    yield ValidationError("{:s} is less than{:s} the minimum value of {:s}.".format(
                        str(instance), (" or equal to" if exclsiveMin else ""), str(minimum)
                    ))

    def _validate_minLength(self, minLength, instance, schema):
        """Validate a minimum length constraint holds for the instance.

        :param minLength:   The minimum length to validate against.
        :type minLength:    int | float
        :param instance:    The schema instance being validated.
        :type instance:     str
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        if self._is_type(minLength, "integer") and minLength >= 0 and self._is_type(instance, "string") and \
                len(instance) < minLength:
            yield ValidationError("{:s} is shorter than the minimum length of {:d}.".format(instance, minLength))

    def _validate_multipleOf(self, multiple, instance, schema):
        """Validate a multiple of constraint holds for the instance.

        :param multiple:    The multiple to validate against.
        :type multiple:     int | float
        :param instance:    The schema instance being validated.
        :type instance:     int | float
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate that the instance is a multiple of the given value.
        if self._is_type(instance, "number"):
            # Only perform validation if the instance is a number.
            if not int(instance % multiple) == (instance % multiple):
                # Validation fails if the instance is not an integer multiple of the given value.
                yield ValidationError("{:s} is not an integer multiple of {:s}.".format(str(instance), str(multiple)))

    def _validate_properties(self, properties, instance, schema):
        """Validate the properties object of the instance.

        :param properties:  The properties object.
        :type properties:   dict
        :param instance:    The schema instance being validated.
        :type instance:     dict
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate the properties.
        if self._is_type(instance, "object"):
            # Only perform validation if the instance is an object.
            for prop, subschema in iteritems(properties):

                # Validate each individual property against its subschema if it appears in the instance being validated.
                if prop in instance:
                    for error in self._validate(instance[prop], subschema):
                        error.path.insert(0, prop)  # Prepend the property to the error path.
                        yield error
                elif subschema.get("required", False):
                    error = ValidationError("{:s} is a required property.".format(prop))
                    error.path.insert(0, prop)  # Prepend the property to the error path.
                    error.validator = "required"
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
    "id": "http://json-schema.org/draft-04/schema#",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Core schema meta-schema",
    "definitions": {
        "schemaArray": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#"}
        },
        "positiveInteger": {
            "type": "integer",
            "minimum": 0
        },
        "positiveIntegerDefault0": {
            "allOf": [ {"$ref": "#/definitions/positiveInteger"}, {"default": 0 } ]
        },
        "simpleTypes": {
            "enum": [ "array", "boolean", "integer", "null", "number", "object", "string" ]
        },
        "stringArray": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True
        }
    },
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "format": "uri"
        },
        "$schema": {
            "type": "string",
            "format": "uri"
        },
        "title": {
            "type": "string"
        },
        "description": {
            "type": "string"
        },
        "default": {},
        "multipleOf": {
            "type": "number",
            "minimum": 0,
            "exclusiveMinimum": True
        },
        "maximum": {
            "type": "number"
        },
        "exclusiveMaximum": {
            "type": "boolean",
            "default": False
        },
        "minimum": {
            "type": "number"
        },
        "exclusiveMinimum": {
            "type": "boolean",
            "default": False
        },
        "maxLength": {"$ref": "#/definitions/positiveInteger"},
        "minLength": {"$ref": "#/definitions/positiveIntegerDefault0"},
        "pattern": {
            "type": "string",
            "format": "regex"
        },
        "additionalItems": {
            "anyOf": [
                {"type": "boolean"},
                {"$ref": "#"}
            ],
            "default": {}
        },
        "items": {
            "anyOf": [
                {"$ref": "#"},
                {"$ref": "#/definitions/schemaArray"}
            ],
            "default": {}
        },
        "maxItems": {"$ref": "#/definitions/positiveInteger"},
        "minItems": {"$ref": "#/definitions/positiveIntegerDefault0"},
        "uniqueItems": {
            "type": "boolean",
            "default": False
        },
        "maxProperties": {"$ref": "#/definitions/positiveInteger"},
        "minProperties": {"$ref": "#/definitions/positiveIntegerDefault0"},
        "required": {"$ref": "#/definitions/stringArray"},
        "additionalProperties": {
            "anyOf": [
                {"type": "boolean"},
                {"$ref": "#"}
            ],
            "default": {}
        },
        "definitions": {
            "type": "object",
            "additionalProperties": {"$ref": "#"},
            "default": {}
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
        "dependencies": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [
                    {"$ref": "#"},
                    {"$ref": "#/definitions/stringArray"}
                ]
            }
        },
        "enum": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True
        },
        "type": {
            "anyOf": [
                {"$ref": "#/definitions/simpleTypes"},
                {
                    "type": "array",
                    "items": {"$ref": "#/definitions/simpleTypes"},
                    "minItems": 1,
                    "uniqueItems": True
                }
            ]
        },
        "allOf": {"$ref": "#/definitions/schemaArray"},
        "anyOf": {"$ref": "#/definitions/schemaArray"},
        "oneOf": {"$ref": "#/definitions/schemaArray"},
        "not": {"$ref": "#"}
    },
    "dependencies": {
        "exclusiveMaximum": [ "maximum" ],
        "exclusiveMinimum": [ "minimum" ]
    },
    "default": {}
}


def main(instance, schema, types=None):
    """

    :param instance:    The JSON instance to validate. This can be either a file location, a JSON object saved as
                        a string or a loaded JSON object.
    :type instance:     str | dict
    :param schema:      The JSON schema to validate against. This can be either a file location, a JSON object saved as
                        a string or a loaded JSON object.
    :type schema:       str | dict
    :param types:       Any additional or alternative types that should be permitted. This can be used to
                        augment or override the default types used. For example, the default number type is an
                        int or float. To change this to include complex numbers set types to
                        {"number": (float, int, complex)}. If you want only integers to be allowed, then set
                        types to {"number": int}.
    :type types:        dict[str, tuple]

    """

    validator = Validator(schema, types if types else {})
    validator.validate_instance(instance)
