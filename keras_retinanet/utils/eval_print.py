"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

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

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

kitti_classes_inv = {
    0: 'Car',
    1: 'Van',
    2: 'Truck',
    3: 'Pedestrian',
    4: 'Person_sitting',
    5: 'Cyclist',
    6: 'Tram',
    7: 'Misc'
}

def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image_name   = os.path.basename(generator.images[i])
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        _print(image_detections, save_path, image_name)

    return all_detections

def _print(detections, save_path, image_name):

    with open(os.path.join(save_path, image_name.replace("png", "txt")), "w") as file:
        for detection in detections:
            x1 = int(round(detection[0]))
            y1 = int(round(detection[1]))
            x2 = int(round(detection[2]))
            y2 = int(round(detection[3]))
            score = detection[4]
            prediction_type = kitti_classes_inv[int(detection[-1])]
            file.write("{} -1 -1 -10 {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {:.2f}\n".format(prediction_type, x1, y1, x2, y2, score * 100))


