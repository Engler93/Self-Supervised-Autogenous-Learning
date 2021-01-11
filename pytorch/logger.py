import tqdm
import sys

BAR_FORMAT = '{desc} {percentage:3.0f}% {elapsed}<{remaining}, {rate_fmt}{postfix}'

class ProgressPrinter(tqdm.tqdm):
    def __call__(self, **kwargs):
        self.set_postfix(refresh=False, **kwargs)
        self.update()

def make_printer(bar_format=BAR_FORMAT, miniters=0,
                 mininterval=0.5, smoothing=0.1, file=sys.stdout, **kwargs):
    """
    Create a ProgressPrinter with appropriate parameters for training.
    See tqdm documentation for details on parameters.
    :return:

    """
    tqdm.tqdm.monitor_interval = 0
    p = ProgressPrinter(bar_format=bar_format, miniters=miniters,
                        mininterval=mininterval, smoothing=smoothing,
                        file=file,
                        **kwargs)
    return p


def print(*args):
    """
    Overwrites the builtin print function with `tqdm.tqdm.write`,
    so things are printed properly while tqdm is active.
    :param args:
    :return:
    """
    return tqdm.tqdm.write(' '.join(map(str, args)), file=sys.stdout)