import os
import argparse

parser = argparse.ArgumentParser(
    epilog="Example: python generate_dataset.py"
)
parser.add_argument(
    "--input-dir",
    help="The column name to add the calculation",
    dest="input_dir",
    type=str,
    required=True
)
args = parser.parse_args()

directory = args.input_dir
hashes = set([fp.split('_')[2] for fp in os.listdir(directory)])
for hash in hashes:
    command = f'python predict.py --train-filepath {directory}/train_dataset_{hash} --test-filepath {directory}/test_dataset_{hash}'
    os.system(command)