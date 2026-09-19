import copy
import logging
import math
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
logger = logging.getLogger(__name__)

class GehringLinear(nn.Linear):

    def __init__(self, in_features, out_features, dropout=0, bias=True, weight_norm=True):
        self.dropout = dropout
        self.weight_norm = weight_norm
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        std = math.sqrt((1 - self.dropout) / self.in_features)
        self.weight.data.normal_(mean=0, std=std)
        if self.bias is not None:
            self.bias.data.fill_(0)
        if self.weight_norm:
            nn.utils.weight_norm(self)

class WNLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool=True, device=None, dtype=None, wnorm=False):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        if wnorm:
            weight_norm(self)
        self.fix_weight_norm_deepcopy()

    def fix_weight_norm_deepcopy(self):
        orig_deepcopy = getattr(self, '__deepcopy__', None)

        def __deepcopy__(self, memo):
            weights = {}
            for hook in self._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    weights[hook.name] = getattr(self, hook.name)
                    delattr(self, hook.name)
            __deepcopy__ = self.__deepcopy__
            if orig_deepcopy:
                self.__deepcopy__ = orig_deepcopy
            else:
                del self.__deepcopy__
            result = copy.deepcopy(self)
            for name, value in weights.items():
                setattr(self, name, value)
            self.__deepcopy__ = __deepcopy__
            return result
        self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)
