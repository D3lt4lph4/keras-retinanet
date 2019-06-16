import argparse

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
        'caravan': 'DontCare',
        'motorcycle': 'DontCare',
        'rider': 'DontCare',
        'bicycle': 'Cyclist',
        'trailer': 'DontCare'
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
            'caravan': 'DontCare',
            'motorcycle': 'DontCare',
            'rider': 'DontCare',
            'bicycle': 'DontCare',
            'trailer': 'DontCare'
        }
    else:
        translation = {
            'car': 'Car',
            'truck': 'Truck',
            'motorcycle': 'Motorcycle'
        }



if args.json:
    json_path = "/save/2017018/bdegue01/datasets/GTA_dataset/bounding_box.json"
    output_path = "/save/2017018/bdegue01/datasets/GTA_dataset/"
    images_directory = "/save/2017018/bdegue01/datasets/GTA_dataset/images"
    
    with open(json_path) as json_file:
        data = json.load(json_file)
    with open(join(output_path, 'boxes_{}.csv'.format(name)), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for image in data:
            filename, file_extension = splitext(image["name"])
            image_path = join(images_directory, filename + "_synthesized_image.jpg")
            for labels in image["labels"]:
                labels["category"] = translation[labels["category"]]
                if labels["category"] == "DontCare":
                    continue
                csv_writer.writerow([image_path, labels["box2d"]["y1"], labels["box2d"]["x1"], labels["box2d"]["y2"], labels["box2d"]["x2"], labels["category"]])
    
else:
    set_path = "/save/2017018/bdegue01/datasets/MIISST_camera_snapshots/sets/val.txt"
    xmls_path = "/save/2017018/bdegue01/datasets/MIISST_camera_snapshots/xmls/"
    output_path = "/save/2017018/bdegue01/datasets/MIISST_camera_snapshots/"
    images_directory = "/save/2017018/bdegue01/datasets/MIISST_camera_snapshots/images"

    with open(set_path) as set_file:
        lines = set_file.read().splitlines()

    with open(join(output_path, 'boxes_val.csv'), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for image_name in lines:
            filename = '{}'.format(image_name) + '.jpg'
            xml = '{}'.format(image_name) + '.xml'
            image_path = join(images_directory, filename)
            xml_path = join(xmls_path, xml)

            tree = ElementTree.parse(xml_path)
            root = tree.getroot()

            for object_tree in root.findall('object'):
                x_min = -1
                y_min = -1
                x_max = -1
                y_max = -1
                
                tag = object_tree.find('name').text
                tag = translation[tag]

                for bounding_box in object_tree.iter('bndbox'):
                    x_min = float(bounding_box.find('xmin').text)
                    y_min = float(bounding_box.find('ymin').text)
                    x_max = float(bounding_box.find('xmax').text)
                    y_max = float(bounding_box.find('ymax').text)

                csv_writer.writerow([image_path, x_min, y_min, x_max, y_max, tag])


with open(join(output_path, 'mapping_{}.csv'.format(name)), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sorted_x = sorted(classes.items(), key=operator.itemgetter(1))
    for key in sorted_x:
        csv_writer.writerow([key[0], key[1]])


        
