import socket
from socket import AddressFamily
from socket import SocketKind

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler, Optimizer

def find_free_port(addr):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((addr, 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def make_init_address(addr=None, port=None):
    if addr is None:
        hostname = addr or socket.gethostname()
        ips = socket.getaddrinfo(
            hostname, 0,
            family=AddressFamily.AF_INET,
            type=SocketKind.SOCK_STREAM
        )
        ips = [i[4][0] for i in ips]
        ips = [i for i in ips if i not in (None, '', 'localhost', '127.0.0.1')]
        if not ips:
            raise RuntimeError('no IPv4 interface found')
        addr = ips[0]
    port = port or find_free_port(addr)
    return 'tcp://%s:%d' % (addr, port)


class Delist(nn.Module):
    def __init__(self, module):
        nn.Module.__init__(self)
        self.module = module

    def forward(self, sample, *args, **kwargs):
        return self.module(sample[0][0])


def _ensure_iterable(v):
    if isinstance(v, str):
        return [v]
    else:
        return list(v)


MEAN = 123.675, 116.28, 103.53
STD = 58.395, 57.12, 57.375


class Normalizer(nn.Module):
    # noinspection PyUnresolvedReferences
    def __init__(self, module, mean=None, std=None, is_float=False):
        nn.Module.__init__(self)
        self.module = module
        if mean is None:
            mean = [123.675, 116.28, 103.53]
        if std is None:
            std = [58.395, 57.12, 57.375]
        if is_float:
            for i in range(3):
                mean[i] = mean[i]/255
                std[i] = std[i]/255
        print(mean)
        print(std)
        self.register_buffer(
            'mean', torch.FloatTensor(mean).view(1, len(mean), 1, 1)/255
        )
        self.register_buffer(
            'std', torch.FloatTensor(std).view(1, len(std), 1, 1)/255
        )

    def forward(self, x):
        x = x.float()  # implicitly convert to float
        x = x.sub(self.mean).div(self.std)
        return self.module(x)


class Unpacker(nn.Module):
    def __init__(self, module, input_key='image', output_key='logits'):
        super(Unpacker, self).__init__()
        self.module = module
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, sample):
        x = sample[self.input_key]
        x = self.module(x)
        sample[self.output_key] = x
        return sample

class _LRPolicy(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        lr_scheduler.LambdaLR.__init__(self, optimizer, 1, last_epoch)
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def step(self, epoch=None, metrics=None):
        lr_scheduler.LambdaLR.step(self, epoch)

class PiecewiseLinear(_LRPolicy):
    def __init__(self, optimizer, knots, vals, last_epoch=-1):
        self.knots = knots
        self.vals = vals
        _LRPolicy.__init__(self, optimizer, last_epoch)
        del self.lr_lambdas

    def get_lr(self):
        r = np.interp([self.last_epoch], self.knots, self.vals)[0]
        return [base_lr * r for base_lr in self.base_lrs]
