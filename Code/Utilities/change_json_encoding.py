"""Code to convert a JSON object from unicode to ascii strings."""

# Python imports.
import operator
import sys

# Standardise variables and methods needed for both Python versions.
if sys.version_info[0] >= 3:
    basestring = unicode = str
    iteritems = operator.methodcaller("items")
else:
    iteritems = operator.methodcaller("iteritems")


def main(jsonObject, encoding="utf-8"):
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
        return dict([(main(key), main(value)) for key, value in iteritems(jsonObject)])
    elif isinstance(jsonObject, list):
        # If the current part of the JSON object is a list, then encode all its elements if needed.
        return [main(i) for i in jsonObject]
    elif isinstance(jsonObject, unicode):
        # If you've reached a unicode string then encode.
        return jsonObject.encode(encoding)
    else:
        # You've reached a non-unicode terminus (e.g. an integer or null).
        return jsonObject