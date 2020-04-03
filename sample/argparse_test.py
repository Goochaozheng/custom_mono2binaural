import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--arg", help='test argument', type=str)
arg = parse.parse_args()

print(type(vars(arg)))

t = {
    'arg_1': 123,
    'arg_2': 123,
    'arg_3': 123
}

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("log")
writer.add_hparams(t, t)


