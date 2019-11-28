"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import os.path

import numpy as np
from PIL import Image
import json

from .generator import Generator
from ..utils.image import read_image_bgr

bdd100k_classes = {
    'bus': 0,
    'traffic light': 1,
    'traffic sign': 2,
    'person': 3,
    'bike': 4,
    'truck': 5,
    'motor': 6,
    'car': 7,
    'train': 8,
    'rider': 9
}

to_select = ["person", "bike", "truck", "..."]


class BDD100KGenerator(Generator):
    """ Generate data for a BDD100K dataset.

    See https://bdd-data.berkeley.edu/ for more information.
    """

    def __init__(
        self,
        base_dir,
        subset='train',
        **kwargs
    ):
        """ Initialize a BDD100K data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            subset: The subset to generate data for (defaults to 'train').
        """
        self.base_dir = base_dir

        labels_file = os.path.join(self.base_dir, "labels", "bdd100k_labels_images_{}.json".format(subset))
        image_dir = os.path.join(self.base_dir, "images", "10k", subset)

        """
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """

        self.labels = {}
        self.classes = bdd100k_classes
        for name, label in self.classes.items():
            self.labels[label] = name
        
        # Load all the labels in a dictionnary
        images_labels = {}
        with open(labels_file) as json_labels:
            data_to_parse = json.load(json_labels)
        
        for data in data_to_parse:
            name = data["name"]
            images_labels[name] = data

        self.image_data = dict()
        self.images = []
        
        for i, fn in enumerate(os.listdir(image_dir)):
            image_fp = os.path.join(image_dir, fn)

            self.images.append(image_fp)

            # Extract label information from the data
            image_data = images_labels[fn]

            boxes = []

            for object_present in image_data["labels"]:
                if object_present["category"] in to_select:
                    cls_id = bdd100k_classes[object_present["category"]]
                    box = object_present["box2d"]
                    x1, x2, y1, y2 = box["x1"], box["x2"], box["y1"], box["y2"],
                    annotation = {'cls_id': cls_id, 'x1': x1, 'x2': x2, 'y2': y2, 'y1': y1}
                    boxes.append(annotation)

            self.image_data[i] = boxes

        super(BDD100KGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.images)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError()

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.images[image_index])
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.images[image_index])

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        image_data = self.image_data[image_index]
        annotations = {'labels': np.empty((len(image_data),)), 'bboxes': np.empty((len(image_data), 4))}

        for idx, ann in enumerate(image_data):
            annotations['bboxes'][idx, 0] = float(ann['x1'])
            annotations['bboxes'][idx, 1] = float(ann['y1'])
            annotations['bboxes'][idx, 2] = float(ann['x2'])
            annotations['bboxes'][idx, 3] = float(ann['y2'])
            annotations['labels'][idx] = int(ann['cls_id'])

        return annotations
