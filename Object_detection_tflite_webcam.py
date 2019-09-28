######## Webcam Object Detection Using Tensorflow-lite trained model #########
# NOTE: This code will work for both - quant and non-quant object detection model
'''
TensorFlow Lite Python interpreter 
Using the interpreter from a model file 
'''

# Import packages
import os
import numpy as np
import tensorflow as tf
import cv2
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Make the type of the trained model as non-quant
floating_model = False

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training','labelmap.pbtxt')

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Number of classes the object detector can identify
NUM_CLASSES = 1

print('PATH_TO_LABELS=', PATH_TO_LABELS)

## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

## Load the trained model
TF_MODEL = os.path.join(CWD_PATH, MODEL_NAME, 'output_tflite_graph.tflite')
interpreter = tf.lite.Interpreter(model_path = TF_MODEL)

interpreter.allocate_tensors()

input_mean=127.5
input_std=127.5

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
if input_details[0]['dtype'] == np.float32:
  floating_model = True

# Get the height and width used while training the model
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]    

print("height=",height)
print("width=",width)

video = cv2.VideoCapture(0)
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Test model on random input data.
input_shape = input_details[0]['shape']

while True:
  ok, image_np = video.read()
  if not ok:
    print('Cannot read video file')
    sys.exit()
  image_np_x = cv2.resize(image_np, (height,width))
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  input_data = np.expand_dims(image_np_x, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  #boxes, scores, classes, num
  boxes = interpreter.get_tensor(output_details[0]['index'])
  classes = interpreter.get_tensor(output_details[1]['index'])
  scores = interpreter.get_tensor(output_details[2]['index'])
  num = int(interpreter.get_tensor(output_details[3]['index'])[0])
  
  if(num > 0):
    print('------------')
    print('num',num)
    for i in range(num):
      classes[0][i]=classes[0][i] + 1.0
      id = int(classes[0][i])
      sco=scores[0][i]
      name='none'
      if(id in category_index):
        s=category_index[id]
        name=s['name']
      print(name,'/',sco)
        
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh= 0.8,       
        max_boxes_to_draw=num,      
        line_thickness=2)
  else:
    print('num', num)

  cv2.imshow('img',image_np)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()
