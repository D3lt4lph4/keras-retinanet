import argparse

import random

from shutil import copy
from os import listdir, makedirs
from os.path import join, basename, splitext

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="The file containing the description of the dataset.") 
parser.add_argument("output_dir", help="The directory where the output should be written")
parser.add_argument("--percent", help="The percentage of the dataset to select", type=float, default=0.1)
args = parser.parse_args()

# Creating the output directories
makedirs(join(args.output_dir, "train", "images"))
makedirs(join(args.output_dir, "train", "labels"))
makedirs(join(args.output_dir, "val", "images"))
makedirs(join(args.output_dir, "val", "labels"))
makedirs(join(args.output_dir, "test", "images"))
makedirs(join(args.output_dir, "test", "labels"))

# Reading all the training files
train_files_images = [join(args.input_dir, "train", "images", f) for f in listdir(join(args.input_dir, "train", "images"))]
val_files_images = [join(args.input_dir, "val", "images", f ) for f in listdir(join(args.input_dir, "val", "images"))]

random.shuffle(train_files_images)

limit = int(len(train_files_images) * args.percent)

test_files_images = train_files_images[limit:]
train_files_images = train_files_images[:limit]

# Copy the training files to the new directory
for train_file in train_files_images:
    filename = basename(train_file)
    root, _ = splitext(filename)

    copy(train_file, join(args.output_dir, "train", "images", filename), follow_symlinks=False)
    copy(train_file, join(args.output_dir, "train", "labels", root + ".txt"), follow_symlinks=False)

# Copy the validation files to the new directory
for val_file in val_files_images:
    filename = basename(val_file)
    root, _ = splitext(filename)

    copy(val_file, join(args.output_dir, "val", "images", filename), follow_symlinks=False)
    copy(val_file, join(args.output_dir, "val", "labels", root + ".txt"), follow_symlinks=False)

# Copy the testing files to the new directory
for test_file in test_files_images:
    filename = basename(test_file)
    root, _ = splitext(filename)

    copy(test_file, join(args.output_dir, "test", "images", filename), follow_symlinks=False)
    copy(test_file, join(args.output_dir, "test", "labels", root + ".txt"), follow_symlinks=False)
