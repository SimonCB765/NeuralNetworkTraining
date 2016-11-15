"""Code to convert a JSON object from unicode to ascii strings."""


def json_to_ascii(jsonObject):
    """Convert a JSON object from unicode to ascii strings.

    This function assumes that the Python version is 2.x (else there is no need to use it).

    This will recurse through all levels of the JSON dictionary, and therefore may hit Python's recursion limit.
    To avoid this use object_hook in the json.load() function instead.

    :param jsonObject:  The JSON object.
    :type jsonObject:   dict
    :return:            The JSON object with all strings converted to ascii format.
    :rtype:             dict

    """

    if isinstance(jsonObject, dict):
        # If the current part of the JSON object is a dictionary, then make all its keys and values ascii if needed.
        return dict([(json_to_ascii(key), json_to_ascii(value)) for key, value in jsonObject.iteritems()])
    elif isinstance(jsonObject, list):
        # If the current part of the JSON object is a list, then make all its elements ascii if needed.
        return [json_to_ascii(i) for i in jsonObject]
    elif isinstance(jsonObject, unicode):
        # If you've reached a unicode string convert it to ascii.
        return jsonObject.encode('utf-8')
    else:
        # You've reached a non-unicode terminus (e.g. an integer or null).
        return jsonObject
