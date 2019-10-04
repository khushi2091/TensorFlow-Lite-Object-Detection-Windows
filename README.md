# TensorFlow-Lite-Object-Detection-Windows
A tutorial showing how to convert, and run TensorFlow Lite object detection models on Windows 10.

This tutorial shows how to test a tensorflow lite object detection model. It is trained using the Object detection tutorial mentioned here https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10. The main objective for this repository is to test the model deployed for rasberry pie or any android application in your computer.

The given code will work for any of the re-trained model on SSD-MobileNet model. I have chosen ssd_mobilenet_v2_coco and ssd_mobilenet_v2_quantized_coco from the given list of pre-trained classfiers among [model zoo]( https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). The code is designed in such a way it will work with both the type of models: quantized as well as non-quantized.

This readme describes every step required to run your tensorflow lite model:
1. Training of Object detectin model (See [Custom object detection](https://github.com/khushi2091/Custom-Object-Detection-Tutorial))
2. Export your trained model using the model check point. Run thefollowing commands: 
  ```
  cd /tensorflow/models/research/object_detetcion/
  python export_tflite_ssd_graph.py --pipeline_config_path=training_hand/pipeline.config --trained_checkpoint_prefix=training_hand/model.ckpt-* --output_directory=training_hand/ --add_postprocessing_op=true
```
It will export the model file (tflite_graph.pb) in the specified training directory
3. Convert your exported ssd mobilenet quantized model into tflite using tflite_convert(See this: https://www.tensorflow.org/lite/convert/cmdline_examples)
```
  cd training_hand/
  tflite_convert --output_file="tflite_graph_hand.tflite" --graph_def_file="tflite_graph.pb" --inference_type=QUANTIZED_UINT8 --input_arrays="normalized_input_image_tensor" --output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" --mean_values=128 --std_dev_values=128 --input_shapes=1,300,300,3 --change_concat_input_ranges=false --allow_nudging_weights_to_use_fast_gemm_kernel=true --allow_custom_ops
```
You will get "tflite_graph_hand.tflite" file inside your folder named as "training_hand"
