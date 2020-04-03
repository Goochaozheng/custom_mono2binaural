import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--arg", help='test argument', type=str)
arg = parse.parse_args()
print(type(vars(arg)))


