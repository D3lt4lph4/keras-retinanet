import argparse

import json

from os.path import join, splitext

import csv

import operator

def extract_xml_information(xml_path):
    results = None

    return results

parser = argparse.ArgumentParser()
parser.argument_default("-json", action="store_true")

args = parser.parse_args()

current_id = 0
id_dict = {}

if args.json:
    json_path = "/save/2017018/bdegue01/datasets/GTA_dataset/bounding_box.json"
    output_path = "/save/2017018/bdegue01/datasets/GTA_dataset/"
    images_directory = "/save/2017018/bdegue01/datasets/GTA_dataset/images"
    
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
    
else:
    set_path = ""
    xmls_path = ""
    output_path = "/save/2017018/bdegue01/datasets/GTA_dataset/"
    images_directory = "/save/2017018/bdegue01/datasets/GTA_dataset/images"
    with open(set_path) as set_file:
        lines = set_file.read().splitlines()

    with open(join(output_path, 'boxes.csv'), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for image_name in lines:
            filename = '{}'.format(image_name) + '.jpg'
            image_path = join(images_dir, filename))

            tree = ElementTree.parse(image_path)
            root = tree.getroot()

            for object_tree in root.findall('object'):
                x_min = -1
                y_min = -1
                x_max = -1
                y_max = -1
                
                tag = object_tree.find('name').text

                if tag not in id_dict:
                    id_dict[tag] = current_id
                    current_id += 1

                for bounding_box in object_tree.iter('bndbox'):
                    x_min = float(bounding_box.find('xmin').text)
                    y_min = float(bounding_box.find('ymin').text)
                    x_max = float(bounding_box.find('xmax').text)
                    y_max = float(bounding_box.find('ymax').text)

                csv_writer.writerow([image_path, x_min, y_min, x_max, y_max, tag])


with open(join(output_path, 'mapping.csv'), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sorted_x = sorted(id_dict.items(), key=operator.itemgetter(1))
    for key in sorted_x:
        csv_writer.writerow([key[0], key[1]])


        
