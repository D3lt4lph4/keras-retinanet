import argparse

import random

import json

from os.path import join, splitext

import csv

import operator

from xml.etree import ElementTree

def extract_xml_information(xml_path):
    results = None

    return results

parser = argparse.ArgumentParser()
parser.add_argument("--json", action="store_true")
parser.add_argument("--kitti", action="store_true")
parser.add_argument("--gta", action="store_true")

args = parser.parse_args()


if args.kitti:
    name = "kitti"
    classes = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': 7
    }
    translation = {
        'truck': 'Truck',
        'person': 'Pedestrian',
        'car': 'Car',
        'bus': 'Truck',
        'caravan': 'Car',
        'motorcycle': 'DontCare',
        'rider': 'DontCare',
        'bicycle': 'Cyclist',
        'trailer': 'Truck'
    }
else:
    name = "miisst"
    classes = {
        'Car': 0,
        'Truck': 1,
        'Motorcycle': 2
    }
    if args.gta:
        translation = {
            'truck': 'Truck',
            'person': 'DontCare',
            'car': 'Car',
            'bus': 'Truck',
            'caravan': 'Car',
            'motorcycle': 'Motorcycle',
            'bicycle': 'DontCare',
            'rider': 'DontCare',
            'trailer': 'Truck'
        }
    else:
        translation = {
            'car': 'Car',
            'truck': 'Truck',
            'motorcycle': 'Motorcycle'
        }



if args.json:
    json_path = "/save/2017018/bdegue01/datasets/GTA_dataset/bounding_box.json"
    output_path = "/save/2017018/bdegue01/datasets/GTA_dataset/datasets"
    images_directory = "/save/2017018/bdegue01/datasets/GTA_dataset/generated_images"
    train_set = "/save/2017018/bdegue01/datasets/GTA_dataset/datasets/boxes_train.txt"
    val_set = "/save/2017018/bdegue01/datasets/GTA_dataset/datasets/boxes_val.txt"

    with open(json_path) as json_file:
        data = json.load(json_file)

    with open(train_set) as train:
        temp = train.readlines()
        train_files = {file.rstrip("\n") for file in temp}

    with open(val_set) as val:
        temp = val.readlines()
        val_files = {file.rstrip("\n") for file in temp}

    csv_file_train = open(join(output_path, 'boxes_{}_train.csv'.format(name)), mode='w')
    csv_writer_train = csv.writer(csv_file_train, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_file_val = open(join(output_path, 'boxes_{}_val.csv'.format(name)), mode='w')
    csv_writer_val = csv.writer(csv_file_val, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for image in data:
        filename, file_extension = splitext(image["name"])
        image_path = join(images_directory, filename + ".jpg")

        labels_written = 0
        for labels in image["labels"]:
            labels["category"] = translation[labels["category"]]
            if labels["category"] == "DontCare":
                continue
            labels_written += 1
            #print(image["name"])
            if image["name"] in val_files:
                csv_writer_val.writerow([image_path, labels["box2d"]["y1"], labels["box2d"]["x1"], labels["box2d"]["y2"], labels["box2d"]["x2"], labels["category"]])
            else:
                csv_writer_train.writerow([image_path, labels["box2d"]["y1"], labels["box2d"]["x1"], labels["box2d"]["y2"], labels["box2d"]["x2"], labels["category"]])

        if labels_written == 0:
            csv_writer_train.writerow([image_path, "", "", "", "", ""])

    csv_file_train.close()
    csv_file_val.close()


else:
    set_path = "/save/2017018/bdegue01/datasets/MIISST_camera_snapshots/sets/test.txt"
    xmls_path = "/save/2017018/bdegue01/datasets/MIISST_camera_snapshots/xmls/"
    output_path = "/save/2017018/bdegue01/datasets/MIISST_camera_snapshots/"
    images_directory = "/save/2017018/bdegue01/datasets/MIISST_camera_snapshots/images"

    with open(set_path) as set_file:
        lines = set_file.read().splitlines()

    with open(join(output_path, 'boxes_test.csv'), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for image_name in lines:
            filename = '{}'.format(image_name) + '.jpg'
            xml = '{}'.format(image_name) + '.xml'
            image_path = join(images_directory, filename)
            xml_path = join(xmls_path, xml)

            tree = ElementTree.parse(xml_path)
            root = tree.getroot()

            if len(root.findall('object')) == 0:
                csv_writer.writerow([image_path, "", "", "", "", ""])

            for object_tree in root.findall('object'):
                x_min = -1
                y_min = -1
                x_max = -1
                y_max = -1
                
                tag = object_tree.find('name').text
                tag = translation[tag]

                for bounding_box in object_tree.iter('bndbox'):
                    x_min = int(bounding_box.find('xmin').text)
                    y_min = int(bounding_box.find('ymin').text)
                    x_max = int(bounding_box.find('xmax').text)
                    y_max = int(bounding_box.find('ymax').text)

                csv_writer.writerow([image_path, x_min, y_min, x_max, y_max, tag])


with open(join(output_path, 'mapping_{}.csv'.format(name)), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sorted_x = sorted(classes.items(), key=operator.itemgetter(1))
    for key in sorted_x:
        csv_writer.writerow([key[0], key[1]])
