import argparse

import random

from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="The file containing the description of the dataset.") 
parser.add_argument("output_dir", help="The directory where the output should be written")
parser.add_argument("--percent", help="The percentage of the dataset to select", type=float, default=0.1)
args = parser.parse_args()

# Reading the data from the dataset file
with open(join(args.input_dir, "train.txt"), "r") as input_dir:
    lines = input_dir.readlines()

random.shuffle(lines)
limit = int(len(lines) * args.percent)
new_test = lines[limit:]
lines = lines[:limit]

# Opening the output file
with open(join(args.output_dir, "train_output.txt"), mode='w') as out_file:
    for image in lines:
        out_file.write(image)

with open(join(args.input_dir, "test.txt"), "r") as input_dir:
    lines_test = input_dir.readlines()
    for line in lines_test:
        new_test.append(line)

# Opening the output file
with open(join(args.output_dir, "test_output.txt"), mode='w') as out_file:
    for image in new_test:
        out_file.write(image)