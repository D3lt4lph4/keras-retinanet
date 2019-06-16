import argparse

import json

from os.path import join, splitext

import csv

import operator

json_path = "/save/2017018/bdegue01/datasets/GTA_dataset/bounding_box.json"
output_path = "/save/2017018/bdegue01/datasets/GTA_dataset/"
images_directory = "/save/2017018/bdegue01/datasets/GTA_dataset/images"
current_id = 0
id_dict = {}

with open(json_path) as json_file:
    data = json.load(json_file)

with open(join(output_path, 'boxes.csv'), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for image in data:
        filename, file_extension = splitext(image["name"])
        image_path = join(images_directory, filename + "_synthesized_image.jpg")
        for labels in image["labels"]:
            if labels["category"] not in id_dict:
                id_dict[labels["category"]] = current_id
                current_id += 1
            csv_writer.writerow([image_path, labels["box2d"]["y1"], labels["box2d"]["x1"], labels["box2d"]["y2"], labels["box2d"]["x2"], labels["category"]])

with open(join(output_path, 'mapping.csv'), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sorted_x = sorted(id_dict.items(), key=operator.itemgetter(1))
    for key in sorted_x:
        csv_writer.writerow([key[0], key[1]])
        
