from __future__ import absolute_import
from . import _external as __external
from ._external import *

__all__ = []
for key in __external.__dict__.keys():
    try:
        __external.__dict__[key].__module__='nifty.external'
    except:
        pass
    __all__.append(key)
