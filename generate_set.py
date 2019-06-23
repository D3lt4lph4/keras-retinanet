import argparse

import random

import json

from os.path import join, splitext

import csv

import operator


def extract_xml_information(xml_path):
    results = None

    return results

parser = argparse.ArgumentParser()
parser.add_argument("--json", action="store_true")
parser.add_argument("--kitti", action="store_true")
parser.add_argument("--gta", action="store_true")

args = parser.parse_args()

json_path = "/save/2017018/bdegue01/datasets/GTA_dataset/bounding_box.json"
output_path = "/save/2017018/bdegue01/datasets/GTA_dataset/datasets"
validation_ratio = 0.05

# Opening the json file
with open(json_path) as json_file:
    data = json.load(json_file)
    random.shuffle(data)
    training_limit = int((1 - validation_ratio) * len(data))


with open(join(output_path, 'boxes_train.txt'), mode='w') as out_file:
    for image in data[:training_limit]:
        out_file.write(image["name"] + "\n")

with open(join(output_path, 'boxes_val.txt'), mode='w') as out_file:
    for image in data[training_limit:]:
        out_file.write(image["name"] + "\n")
