__version__ = '0.6.0'

from abc import ABC

class SKMapBase(ABC):

  def _verbose(self, *args, **kwargs):
    if self.verbose:
      ttprint(*args, **kwargs)