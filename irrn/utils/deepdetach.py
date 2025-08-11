import types
import weakref

import torch as th
from torch import nn


class NonDetachableFlag:
    pass


def NonDetachable(obj):
    """
    This function is used to wrap objects with non detachable flag in order for them not to be detached by deep_detach.

    :param obj: object to wrap
    """
    if isinstance(obj, (type(None), int, float, bool, complex, str,
                        bytes, type, range, slice,
                        types.BuiltinFunctionType, type(Ellipsis), type(NotImplemented),
                        types.FunctionType, weakref.ref)) or issubclass(type(obj), type):
        return obj

    class NonDetachableObj(obj.__class__, NonDetachableFlag):
        pass

    result = NonDetachableObj()

    if isinstance(obj, (list, tuple, dict, frozenset, bytearray, set)):
        return NonDetachableObj(obj)

    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            try:
                setattr(result, k, v)
            except TypeError:
                pass
    return result


def deep_detach(obj, memo=None):
    """
    This function is used in recursive way analogously to deepcopy and allows to create objects with detached tensors.
    It does not consume additional memory, since all non-tensor objects are returned as is.

    :param obj: object to process
    :param memo: dict with already processed objects
    :return: new object with detached tensors (if there are any inside)
    """
    if isinstance(obj, NonDetachableFlag):
        return obj
    if memo is None:
        memo = {}
    if id(obj) in memo:
        return memo[id(obj)]

    if isinstance(obj, th.Tensor):
        requires_grad_tag = obj.requires_grad
        if requires_grad_tag:
            result = obj.detach()
            result.requires_grad_(requires_grad_tag)
            if isinstance(obj, nn.Parameter):
                result = nn.Parameter(result)
        else:
            result = obj
        memo[id(obj)] = result
        return result
    elif isinstance(obj, (type(None), int, float, bool, complex, str,
                          bytes, type, range, slice,
                          types.BuiltinFunctionType, type(Ellipsis), type(NotImplemented),
                          types.FunctionType, weakref.ref)) or issubclass(type(obj), type):
        result = obj
        memo[id(obj)] = result
    else:
        cls = obj.__class__
        try:
            result = cls.__new__(cls)
        except TypeError:
            result = obj

        memo[id(obj)] = result
        if isinstance(obj, list):
            for elem in obj:
                result.append(deep_detach(elem, memo))
        elif isinstance(obj, tuple):
            new_list = []
            for elem in obj:
                new_list.append(deep_detach(elem, memo))
            result = tuple(new_list)
            memo[id(obj)] = result
        elif isinstance(obj, dict):
            for dict_key, dict_value in obj.items():
                result[deep_detach(dict_key, memo)] = deep_detach(dict_value, memo)
        elif isinstance(obj, bytearray):
            result = obj
            memo[id(obj)] = result
        elif isinstance(obj, frozenset):
            result = set()
            for v in obj:
                result.add(deep_detach(v))
            result = frozenset(result)
            memo[id(obj)] = result
        elif isinstance(obj, set):
            for v in obj:
                result.add(deep_detach(v))

    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            try:
                setattr(result, k, deep_detach(v, memo))
            except TypeError:
                pass
    return result
