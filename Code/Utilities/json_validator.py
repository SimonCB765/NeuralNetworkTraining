"""
A simpkified JSON schema draft 4 validator for Python (with optional unicode to ascii string conversion). See the
following links for information on JSON schema:
    http://json-schema.org/
    https://spacetelescope.github.io/understanding-json-schema/
    https://spacetelescope.github.io/understanding-json-schema/UnderstandingJSONSchema.pdf

As per the JSON schema, keywords only limit the range of values of certain primitive types. For example,
"the 'maxLength' keyword will only restrict certain strings (that are too long) from being valid. If the instance is a
number, boolean, null, array, or object, the keyword passes validation". Additionally, validation keywords that are
missing do not prevent validation occurring. The validation therefore permits everything except when validation
explicitly fails.

The simplest way to validate a JSON instance against a schema is to call the :func:`main` function.

The missing components of draft 4 are:
    The required property is not checked for valid formatting. Required properties are still enforced.
    Pattern properties are not included.
    Additional properties are not included.

"""

# Python imports.
import json
import operator
import os
import re
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

        # Add additional types.
        self._types = dict(self.defaultTypes)
        if types:
            types = {i: tuple(j) for i, j in iteritems(types)}
            self._types.update(types)
        self._types["any"] = tuple(self._types.values())  # Generate the record for an any type validator.

        # Validate the schema against the saved meta-schema if desired.
        if checkValidity:
            self.validate_instance(self._schema, self.metaSchema)

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

    def _validate_additionalItems(self, addItems, instance, schema):
        """Validate additional items in an instance of an items definition against the schema it should match.

        :param addItems:    The additional item definition being validated against.
        :type addItems:     dict | bool
        :param instance:    The schema instance being validated.
        :type instance:     list
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Ensure we are only validating lists.
        if not self._is_type(instance, "array"):
            return

        # Additional items only matter when the items schema definition is an array of schemas, as otherwise you can
        # make the array of items as long as you want.
        if not self._is_type(schema.get("items"), "array"):
            return

        # Validate the additional items.
        if self._is_type(addItems, "object"):
            for item in instance[len(instance[len(schema.get("items", [])):])]:
                for error in self._validate(addItems, item):
                    yield error
        elif self._is_type(addItems, "boolean") and not addItems and len(instance) > len(schema.get("items", [])):
            # If additionalItems is a boolean, then we are only interested in validating when it is false (as if it is
            # true you can just add as many additional items as you want). If additionalItems is false and the length
            # of the array of items is longer than the schema definition, you have a validation error.
            yield ValidationError("Additional items {:s} not allowed.".format(
                ', '.join(instance[len(schema.get("items", [])):])
            ))

    def validate_dependencies(self, dependencies, instance, schema):
        """Validate the dependencies of an instance.

        :param dependencies:    The dependencies definition being validated against.
        :type dependencies:     dict
        :param instance:        The schema instance being validated.
        :type instance:
        :param schema:          The schema the instance is being validated against.
        :type schema:           dict

        """

        # Only validate objects.
        if not self._is_type(instance, "object"):
            return

        # Validate the dependencies.
        for prop, dependency in iteritems(dependencies):
            if prop not in instance:
                # Only care about dependencies when the property is present.
                continue

            # Validate an individual dependency.
            if self._is_type(dependency, "object"):
                # The dependency is a schema dependency.
                for error in self._validate(instance, dependency):
                    yield error
            elif self._is_type(dependency, "array"):
                # The dependency is a property dependency.
                dependency = [dependency] if isinstance(dependency, basestring) else dependency
                for i in dependency:
                    if i not in instance:
                        yield ValidationError("{:s} is a missing dependency of {:s}.".format(i, prop))

    def _validate_items(self, items, instance, schema):
        """Validate an instance of an items definition against the schema it should match.

        :param items:       The item definition being validated against.
        :type items:        dict | list[dict]
        :param instance:    The schema instance being validated.
        :type instance:     list
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Ensure we are only validating lists.
        if not self._is_type(instance, "array"):
            return

        # Validate the items instance.
        if self._is_type(items, "object"):
            # Validate each item in the instance against the single schema for the items.
            for index, item in enumerate(instance):
                for error in self._validate(items, items):
                    error.path.insert(0, index)
                    yield error
        elif self._is_type(items, "array"):
            # Validate each item in the instance against the array of schemas for the items.
            for (index, item), subschema in zip(enumerate(instance), items):
                for error in self._validate(item, subschema):
                    error.path.insert(0, index)
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

    def _validate_maxItems(self, maxItems, instance, schema):
        """Validate a maximum number of items constraint holds for the instance.

        :param maxItems:    The maximum number of items to validate against.
        :type maxItems:     int
        :param instance:    The schema instance being validated.
        :type instance:     str
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate the number of items.
        if self._is_type(maxItems, "integer") and maxItems >= 0 and self._is_type(instance, "array") and \
                len(instance) >= maxItems:
            yield ValidationError("{:s} has more items than the maximum of {:d}.".format(str(instance), maxItems))

    def _validate_maxLength(self, maxLength, instance, schema):
        """Validate a maximum length constraint holds for the instance.

        :param maxLength:   The maximum length to validate against.
        :type maxLength:    int
        :param instance:    The schema instance being validated.
        :type instance:     str
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate length.
        if self._is_type(maxLength, "integer") and maxLength >= 0 and self._is_type(instance, "string") and \
                len(instance) > maxLength:
            yield ValidationError("{:s} is longer than the maximum length of {:d}.".format(instance, maxLength))

    def _validate_maxProperies(self, maxProp, instance, schema):
        """Validate that an instance has no more than a maximum number of properties.

        :param maxProp:     The maximum number of properties to validate against.
        :type maxProp:      int
        :param instance:    The schema instance being validated.
        :type instance:     dict
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate the maximum number of properties.
        if self._is_type(instance, "object"):
            # Only validate instances that are objects.
            if self._is_type(maxProp, "integer") and maxProp >= 0:
                # Only validate when the maximum number is an integer greater than or equal to 0.
                if len(instance) > maxProp:
                    yield ValidationError(
                        "The number of properties {:d} was greater than the maximum allowed {:d}".format(
                            len(instance), maxProp
                        ))

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

    def _validate_minItems(self, minItems, instance, schema):
        """Validate a minimum number of items constraint holds for the instance.

        :param minItems:    The minimum number of items to validate against.
        :type minItems:     int
        :param instance:    The schema instance being validated.
        :type instance:     str
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate the number of items.
        if self._is_type(minItems, "integer") and minItems >= 0 and self._is_type(instance, "array") and \
                len(instance) <= minItems:
            yield ValidationError("{:s} has fewer items than the minimum of {:d}.".format(str(instance), minItems))

    def _validate_minLength(self, minLength, instance, schema):
        """Validate a minimum length constraint holds for the instance.

        :param minLength:   The minimum length to validate against.
        :type minLength:    int
        :param instance:    The schema instance being validated.
        :type instance:     str
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate length.
        if self._is_type(minLength, "integer") and minLength >= 0 and self._is_type(instance, "string") and \
                len(instance) < minLength:
            yield ValidationError("{:s} is shorter than the minimum length of {:d}.".format(instance, minLength))

    def _validate_minProperies(self, minProp, instance, schema):
        """Validate that an instance has no fewer than a minimum number of properties.

        :param minProp:     The minimum number of properties to validate against.
        :type minProp:      int
        :param instance:    The schema instance being validated.
        :type instance:     dict
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate the minimum number of properties.
        if self._is_type(instance, "object"):
            # Only validate instances that are objects.
            if self._is_type(minProp, "integer") and minProp >= 0:
                # Only validate when the minimum number is an integer greater than or equal to 0.
                if len(instance) < minProp:
                    yield ValidationError("The number of properties {:d} was less than the minimum allowed {:d}".format(
                        len(instance), minProp
                    ))

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

    def _validate_pattern(self, pattern, instance, schema):
        """Validate an instance against a pattern.

        :param pattern:     The pattern to validate against.
        :type pattern:      str
        :param instance:    The schema instance being validated.
        :type instance:     str
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate that the instance is an instance f the given pattern.
        if self._is_type(pattern, "string") and self._is_type(instance, "string") and not re.match(pattern, instance):
            yield ValidationError("{:s} does not match the pattern {:s}.".format(instance, pattern))

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

    def _validate_uniqueItems(self, isUnique, instance, schema):
        """Validate that a unique items constraint holds for the instance.

        This check assumes that the items are hashable and can therefore be turned into a set.

        :param isUnique:    Whether unique items are being enforced.
        :type isUnique:     bool
        :param instance:    The schema instance being validated.
        :type instance:     list
        :param schema:      The schema the instance is being validated against.
        :type schema:       dict

        """

        # Validate that all items are unique.
        if self._is_type(isUnique, "boolean") and isUnique and self._is_type(instance, "array") and \
                (len(set(instance)) == len(instance)):
            yield ValidationError("{:s} has non-unique elements.".format(str(instance)))

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
    """Perform validation of a schema instance against the schema it should follow.

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
