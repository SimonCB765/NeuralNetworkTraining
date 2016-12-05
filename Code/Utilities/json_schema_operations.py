"""Code for manipulating JSON schemas."""

# Python imports.
import operator
import sys

# Define functions for compatibility.
if sys.version_info[0] >= 3:
    from functools import reduce
    basestring = unicode = str
    iteritems = operator.methodcaller("items")
else:
    iteritems = operator.methodcaller("iteritems")


def change_encoding(jsonObject, encoding="utf-8"):
    """Convert unicode strings in a JSON object to a given encoding.

    The primary purpose of this is to convert unicode strings generated in Python 2 to ascii (utf-8)

    This will recurse through all levels of the JSON dictionary, and therefore may hit Python's recursion limit.
    To avoid this use object_hook in the json.load() function instead.

    :param jsonObject:  The JSON object.
    :type jsonObject:   dict
    :param encoding:    The encoding to use.
    :type encoding:     str
    :return:            The JSON object with all strings encoded as desired.
    :rtype:             dict

    """

    if isinstance(jsonObject, dict):
        # If the current part of the JSON object is a dictionary, then encode all its keys and values if needed.
        return dict(
            [(change_encoding(key, encoding), change_encoding(value, encoding)) for key, value in iteritems(jsonObject)]
        )
    elif isinstance(jsonObject, list):
        # If the current part of the JSON object is a list, then encode all its elements if needed.
        return [change_encoding(i, encoding) for i in jsonObject]
    elif isinstance(jsonObject, unicode):
        # If you've reached a unicode string then encode.
        return jsonObject.encode(encoding)
    else:
        # You've reached a non-unicode terminus (e.g. an integer or null).
        return jsonObject


def extract_schema_defaults(schema):
    """Extract the default attribute values derived from the types and defaults specified in a JSON schema.

    If no user supplied default is present in the schema for a given element, then only a "null" element will get a
    default defined for it (a default of None). Only elements with defaults set (either on themselves or a sub-schema)
    will have defaults returned.

    Points to note are:
        - Validation of the schema structure is not performed. The schema should therefore be validated first.
        - Validation of the default values is not performed. It is up to the schema writer to ensure they are legal.
        - References are all treated as references to top level elements of the schema outside the "properties" element.
            Traditionally the referenced elements would be held in a top level "definitions" object.
        - If a $ref keyword appears in an object, then the referenced element is imported into the space of the object.
            For example, if there is a referenced object
                {"description": "...", "type": "object", "ReffedKey1": ..., "ReffedKey2": ...}
            at path "#/path/to/ref", then the object
                {"$ref": "#/path/to/ref"}
            will get converted to
                {"description": "...", "type": "object", "ReffedKey1": ..., "ReffedKey2": ...}
            Any clashes in key names will be decided by overwriting the original value associated with the key with the
            value from the referenced object.
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
    if propsType == "object":
        # This (sub-)schema is an object. If no default is specified, we set a temporary default of an empty dictionary.
        # This temporary default value is never propagated up to a parent schema. A default can only be propagated up
        # if this (sub-)schema has a default or one of its sub-schemas has a default.
        defaults = schemaProps.get("default", None)
        if defaults is not None:
            defaultsFound = True
        defaults = {}
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
                #       "$ref": "#/definitions/DefLocation"
                #   }
                # In this case, rather than going through each element in the EleWithRef object, we directly
                # replace the {"$ref": "#/definitions/DefLocation"} object with the one located at
                # #/definitions/DefLocation and go through that instead. We also directly incorporate the element at
                # #/definitions/DefLocation into the CurrentSchema element due to the
                # "$ref": "#/definitions/DefLocation" element.
                defPath = schemaProps[i].get("$ref").split("/")[1:]  # Ignore the '#' at the beginning of the ref path.
                del schemaProps[i]["$ref"]
                referencedProps = reduce(lambda d, key: d.get(key) if d else None, defPath, schema)
                schemaProps[i].update(referencedProps)
            except AttributeError:
                # The element does not contain a reference (and may not even be an object), so there is no need to
                # manipulate the element.
                pass
            subschema["properties"] = schemaProps[i]
            subschemaDefaults, defaultExtracted = extract_schema_defaults(subschema)
            if defaultExtracted:
                # A sub-schema has a default, so there is a default value for this (sub-)schema.
                defaultsFound = True
                defaults[i] = subschemaDefaults
    elif propsType in ["array", "boolean", "integer", "number", "string"]:
        # The element is a basic type, so we just need to try and extract a default value.
        defaults = schemaProps.get("default", None)
        if defaults is not None:
            defaultsFound = True
    elif propsType == "null":
        # The element is a null type, so just leave the default as None.
        defaultsFound = True
        defaults = None

    return defaults, defaultsFound
