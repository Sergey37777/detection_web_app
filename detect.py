import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
from object_detection.utils import visualization_utils as viz_utils
import os
import json

def detect_objects(detect_fn, frame, category_index):
    def load_image_into_numpy_array(path):
        """Load an image from file into a numpy array.
        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.
        Args:
          path: the file path to the image
        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(path))


    image = frame
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=0.5,
          agnostic_mode=False)
    count = 0
    for root_dir, cur_dir, files in os.walk(r'images'):
      count = len(files)
    cv2.imwrite(f'images/{count}.jpg', image_with_detections)
    # cv2.imshow("image", image_swith_detections)
    # print(detections['detection_boxes'])
    return "Done" #json.dumps(detections['detection_boxes'])
    # DISPLAYS OUTPUT IMAGEs