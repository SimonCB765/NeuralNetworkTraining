"""Code to extract default parameters from a JSON schema."""

# Python imports.
import sys

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    from functools import reduce


def main(schema):
    """Extract the default attribute values derived from the types and defaults specified in a JSON schema.

    If no user supplied default is present in the schema for a given element, then defaults are defined as:
        "array" - []
        "object" - {}
        "null" - None
        "boolean", "integer", "number", "string" - No base default exists.
    Any boolean, integer, number or string valued element that does not have a default defined for it will be left out
    of the defaults returned. For example, the schema:
        {
            "type": "object",

            "arrayElement": {"type": "array", ...},
            "intElement": {"type": "integer"}
        }
    will return the default dictionary:
        {"arrayElement": []}
    As "intElement" has no default defined for it (and an integer element can not be given a base default value), the
    element gets no default and is left out of the returned default dictionary. If this element is not given a value
    in the instantiation of the schema, then it will not appear anywhere in either the instantiated object or the
    schema defaults.

    Points to note are:
        - The base default values are all Falsey.
        - Validation of the schema structure is not performed. The schema should therefore be validated first.
        - Validation of the default values is not performed. It is up to the schema writer to ensure they are legal.
        - References are all treated as references to top level elements of the schema outside the "properties" element.
            Traditionally the referenced elements would be held in a top level "definitions" object.
        - If a $ref keyword appears in an object, then the entire object is overwritten with the referenced element.
            For example, both the object
                {"$ref": "#/path/to/ref"}
            and
                {"$ref": "#/path/to/ref", "otherKey": "otherVal", "extraObject": {...}}
            will be replaced by the object referenced by "#/path/to/ref". The "otherKey" and "extraObject" keys/values
            will be ignored.
        - Defaults specified in a sub-schema will override those specified higher up the schema hierarchy. For example,
            {
                "default": {"key": 0},
                "type": "object",
                "key": {"default": 1, "type": "integer"}
            }
            will produce a default of {"key": 1} not {"key": 0}.
        - There is no need to set a default for a required element of a (sub-)schema (although there is no requirement
            not to) as a value has to be supplied in each instantiation of the schema for the instantiation to
            validate successfully.

    :param schema:  A JSON object containing a schema that the instantiations will be evaluated against.
    :type schema:   dict
    :return:        The default values for the current (sub-)schema.
    :rtype:         dict

    """

    # The returned results for this (sub-)schema. Two variables are needed as every Falsey value is a possible valid
    # default value that can be returned. It's therefore not possible to simple test the returned defaults value for
    # truthiness in order to determine whether default values were found. Nor is it possible to test for None
    # explicitly as the default value for a "null" type element is None, so this is also a valid default value.
    defaults = None
    defaultsFound = False

    # Determine the type of the current (sub-)schema.
    schemaProps = schema.get("properties", {})
    try:
        # If no type is defined then we default to "object". This only can occur at the top-level of the schema when
        # we are first looking at the real original "properties" value (as this is implicitly an object and does not
        # have a type stored within it).
        propsType = schemaProps.get("type", "object")
    except AttributeError:
        # The only elements that do not have a recorded type are either invalid objects where the type was forgotten
        # (which would be caught by validating the schema) or entries like the "description" field of an object. When
        # examining these fields we just want to skip them as they're of no interest in setting defaults.
        propsType = None

    # Extract defaults for the (sub-)schema.
    if propsType == "array":
        # This (sub-)schema is an array. Therefore, we'll set its default value as an empty list if no default
        # is specified.
        defaults = schemaProps.get("default", [])
        defaultsFound = True
    elif propsType == "object":
        # This (sub-)schema is an object. Therefore, we'll set its default value as an empty dictionary if no
        # default is specified, and then look at the elements it contains.
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
                defPath = schemaProps[i].get("$ref").split("/")[1:]  # Ignore the '#' at the beginning of the ref path.
                subschema["properties"] = reduce(lambda d, key: d.get(key) if d else None, defPath, schema)
            except AttributeError:
                # The element does not contain a reference (and may not even be an object), so we just look directly
                # at the element.
                subschema["properties"] = schemaProps[i]
            subschemaDefaults, defaultExtracted = main(subschema)
            if defaultExtracted:
                defaults[i] = subschemaDefaults
    elif propsType in ["boolean", "integer", "number", "string"]:
        # The element is a basic type, so we just need to try and extract a default value.
        defaults = schemaProps.get("default", None)
        if defaults is not None:
            defaultsFound = True
    elif propsType == "null":
        # The element is a null type, so just leave the default as None.
        defaultsFound = True
        defaults = None

    return defaults, defaultsFound
