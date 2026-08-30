"""Small shared validation helpers for Pepsy's public index boundaries."""

from numbers import Integral


def is_strict_integer(value):
    """Return whether ``value`` is an integer value but not a boolean."""

    return isinstance(value, Integral) and not isinstance(value, bool)


def normalize_integer(value, *, name):
    """Return one validated Python integer without lossy coercion."""

    if not is_strict_integer(value):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def normalize_integer_tuple(values, *, name, allow_scalar=True):
    """Return a tuple of strict integers, optionally accepting one scalar."""

    if allow_scalar and is_strict_integer(values):
        return (int(values),)
    try:
        values = tuple(values)
    except TypeError as exc:
        expected = "an integer or an iterable of integers" if allow_scalar else "an iterable of integers"
        raise TypeError(f"{name} must be {expected}.") from exc
    if not all(is_strict_integer(value) for value in values):
        raise TypeError(f"{name} must contain only integers.")
    return tuple(int(value) for value in values)
