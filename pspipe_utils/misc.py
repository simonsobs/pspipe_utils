"""
Some general utility functions that do not belong to other files.
"""


def str_replace(my_str, old, new):
    my_new_str = my_str
    my_new_str = my_str.replace(old, new)
    if my_new_str == my_str:
        error = f" the name '{my_str}' does not contain '{old}' so I can't replace '{old}' by '{new}'"
        raise NameError(error)
    return my_new_str
