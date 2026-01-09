def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def is_str(value):
    return isinstance(value, str)


def is_list(value):
    return isinstance(value, list)


def is_dict(value):
    return isinstance(value, dict)


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def unique_list(input_list):
    return list(set(input_list))
