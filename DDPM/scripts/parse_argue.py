import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--train",
    help = "training",
    action = "store_true",
)

parser.add_argument(
    "--grid",
    help = "generate grid_sample",
    action = "store_true",
)

parser.add_argument(
    "--sample",
    help = "sampling",
    action = "store_true",
)

parser.add_argument(
    "--epoch",
    dest = "epoch_idx",
)