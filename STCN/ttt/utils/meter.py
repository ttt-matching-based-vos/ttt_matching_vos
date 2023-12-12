import torch
import numpy as np

from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.3f'):
        self.name = name
        self.fmt = fmt
        self.values = []

    def reset(self):
        self.values = []

    def update(self, val):
        self.values.append(val)

    def avg(self, n=None):
        arr = self.values[-n:] if n is not None else self.values
        return np.mean(arr)

    def std(self, n=None):
        arr = self.values[-n:] if n is not None else self.values
        return np.std(arr)

    def get_stats(self, n=None):
        return self.avg(n), self.std(n)

    def last(self):
        return self.values[-1]

    def __len__(self):
        return len(self.values)

    def __str__(self):
        fmtstr = '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(val=self.last(), avg=self.avg())


class AverageMeterDict(object):
    def __init__(self):
        self.meter_dict = OrderedDict()

    def reset(self):
        for k, v in self.meter_dict.items():
            v.reset()

    def add(self, name, fmt=':.3f'):
        self.meter_dict[name] = AverageMeter(name, fmt)

    def get(self, name):
        return self.meter_dict[name]

    def update(self, name, val, fmt=':.3f'):
        if isinstance(val, torch.Tensor):
            val = val.clone().detach().cpu().numpy()
        if name not in self.meter_dict:
            self.add(name, fmt)
        self.meter_dict[name].update(val)

    def avg(self, n=None):
        return {k: v.avg(n) for k, v in self.meter_dict.items()}

    def last(self):
        return {k: v.last() for k, v in self.meter_dict.items()}

    def items(self):
        return self.meter_dict.items()

    def keys(self):
        return self.meter_dict.keys()

    def values(self):
        return self.meter_dict.values()

    def values_avg(self):
        return [v.avg() for v in self.meter_dict.values()]

    def to_str(self):
        return {k: str(v) for k, v in self.meter_dict.items()}

    def print(self, delimiter=' '):
        return delimiter.join(['{}: {}'.format(k, str(v)) for k, v in self.meter_dict.items()])

    def __len__(self):
        return min([len(v) for v in self.meter_dict.values()]) if len(self.meter_dict) else 0
