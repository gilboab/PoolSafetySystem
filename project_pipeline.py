
# coding: utf-8


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import matplotlib.patches as patches

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# ## Env setup

# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 


# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[ ]:

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

#PATH_TO_TEST_IMAGES_DIR = 'C:\\CV\\tmp\\models\\research\\object_detection\\Pool_Images'
PATH_TO_TEST_IMAGES_DIR = 'C:\\CV\\tmp\\models\\research\\object_detection\\My_Pool'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'img1.jpg') ]
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'pool{}.jpg'.format(i)) for i in range(22, 23) ]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def find_pool_poligon (frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV - 90 - 110 is the pool color you can tweak it
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blur = cv2.GaussianBlur(hsv[:,:,0],(5,5),0)
    edges = cv2.Canny(blur,100,200,3)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #cv2.imshow('mask', mask)
    img = mask
    # threshold image
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    #cv2.imshow('Thresh', thresh)
    # find contours
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest contour
    i=0
    largest_area=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > largest_area):
            largest_area = area
            largest_contour_index = i
        i=i+1
    hull = cv2.convexHull(contours[largest_contour_index])
    new_frame = frame.copy()
    for i in range (0,np.shape(frame)[0]):
        for j in range (0,np.shape(frame)[1]):
            if cv2.pointPolygonTest(hull, (j,i), False )<0:
                new_frame[i,j,:] = (0,0,0)
    #cv2.drawContours(new_frame, [hull], 0, (0, 0, 255), 3)
    hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest contour
    i=0
    largest_area=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > largest_area):
            largest_area = area
            largest_contour_index = i
        i=i+1
    #cv2.imshow('mask',mask)

    pct = 0.005
    epsilon = pct * cv2.arcLength(contours[largest_contour_index], True)
    approx = cv2.approxPolyDP(contours[largest_contour_index], epsilon, True)
    x = len(approx)
    while x>200:
        epsilon = pct * cv2.arcLength(contours[largest_contour_index], True)
        approx = cv2.approxPolyDP(contours[largest_contour_index], epsilon, True)
        x=len(approx)
        pct=pct+0.005

    out = frame.copy()
    hull_img = frame.copy()
    hull = cv2.convexHull(approx)
    return hull

def plot_img_with_bbox(im, bbox_in, bbox_out, title_text = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(bbox_in.shape[0]):
        ax.add_patch(
            patches.Rectangle(
                (bbox_in[i,0], bbox_in[i,1]),
                bbox_in[i,2],
                bbox_in[i,3],
                fill=False,
                edgecolor='red'
            )
        )
    for i in range(bbox_out.shape[0]):
        ax.add_patch(
            patches.Rectangle(
                (bbox_out[i,0], bbox_out[i,1]),
                bbox_out[i,2],
                bbox_out[i,3],
                fill=False,
                edgecolor='blue'
            )
        )
    plt.imshow(im)
    if title_text is not None:
        plt.title(title_text)


for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  pool_poligon = find_pool_poligon (image_np)
  new_img = image_np.copy()
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  hull_img = image_np.copy()
  #print ('detection boxes')
  bbox = np.all([output_dict['detection_boxes'][:,0]>0],axis=0)
  norm_bbox  = output_dict['detection_boxes'][bbox,:]
  #convert boxes from ymin, xmin, ymax, xmax to xmin, ymin, width, height and unmnormalize
  bboxes = np.zeros((np.shape(norm_bbox)[0],4))
  bboxes[:,0] = norm_bbox[:,1]*np.shape(new_img)[1]
  bboxes[:,1] = norm_bbox[:,0]*np.shape(new_img)[0]
  bboxes[:,2] = (norm_bbox[:,3]-norm_bbox[:,1])*np.shape(new_img)[1]
  bboxes[:,3] = (norm_bbox[:,2]-norm_bbox[:,0])*np.shape(new_img)[0]
  #plt.figure(figsize=IMAGE_SIZE)
  #cv2.imwrite('./Pool_Images/pool22_detect_m3.jpg',image_np)
  #plt.imshow(image_np)
  #plt.imshow(hull_img)
  #cv2.imshow('lll',hull_img)
  box_in_pool = np.zeros((1,4))
  box_not_in_pool = np.zeros((1,4))
  for i in range (0,np.shape(bboxes)[0]):
      left_corner = cv2.pointPolygonTest(pool_poligon, (bboxes[i,0],bboxes[i,1]+bboxes[i,3]), False )
      right_corner = cv2.pointPolygonTest(pool_poligon, (bboxes[i,0]+bboxes[i,2],bboxes[i,1]+bboxes[i,3]), False )
      print (left_corner, right_corner)
      if left_corner>=0 or right_corner>=0:
          box_in_pool = np.append(box_in_pool, np.reshape(bboxes[i,0:4],(1,4)),axis=0)
      else:
          box_not_in_pool = np.append(box_not_in_pool, np.reshape(bboxes[i,0:4],(1,4)),axis=0)
          
  print (bboxes)
  print (box_in_pool)
  print (pool_poligon)
  box_in_pool = box_in_pool[1:np.shape(box_in_pool)[0],:]
  box_not_in_pool = box_not_in_pool[1:np.shape(box_not_in_pool)[0],:]
  cv2.drawContours(new_img, [pool_poligon], 0, (0, 0, 255), 3)
  plot_img_with_bbox (new_img, box_in_pool, box_not_in_pool) 
  plt.show()

