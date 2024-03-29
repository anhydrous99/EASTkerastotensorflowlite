import os
import argparse
import numpy as np
import tensorflow as tf
from freeze import freeze_session
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K

K.set_learning_phase(0)

parser = argparse.ArgumentParser(
    description='Converts Keras Model to Tensorflow Model'
)
parser.add_argument(
    '-j', '--keras_json',
    required=True,
    help='Keras json model'
)
parser.add_argument(
    '-k', '--keras_data',
    required=True,
    help='Keras model'
)
parser.add_argument(
    '-i', '--input_size',
    default=224,
    help='Size of input images (square image)'
)
parser.add_argument(
    '-r', '--resize_factor',
    default=2,
    help='The resize factor set in the EAST model'
)
parser.add_argument(
    '-n', '--name',
    required=True,
    help='Model Name'
)
args = parser.parse_args()

model = model_from_json(open(args.keras_json).read(), custom_objects={'tf': tf, 'RESIZE_FACTOR': args.resize_factor})
model.load_weights(args.keras_data)

input_arrays = ['input_image']

output_names = [out.op.name for out in model.outputs]
new_output_names = ['pred_score_map/Tanh', 'rbox_geo_map/Tanh', 'rbox_angle_map/Tanh']
print(output_names)

frozen_graph = freeze_session(K.get_session(), output_names=output_names)

tf.io.write_graph(frozen_graph, args.name, 'frozen_inference_graph.pb', as_text=False)
tf_model_path = os.path.join(args.name, 'frozen_inference_graph.pb')

converter = tf.lite.TFLiteConverter.from_frozen_graph(tf_model_path, input_arrays, new_output_names,
                                                      input_shapes={
                                                          'input_image': [1, args.input_size, args.input_size, 3]})
converter.inference_type = tf.uint8
converter.quantized_input_stats = {'input_image': (128, 127)}
converter.default_ranges_stats = (0, 6)

tflite_model = converter.convert()
tflite_model_path = os.path.join(args.name, args.name + '.tflite')
open(tflite_model_path, 'wb').write(tflite_model)

# Test tensorflow lite
# Load TFLite model and alloate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data1 = interpreter.get_tensor(output_details[0]['index'])
output_data2 = interpreter.get_tensor(output_details[1]['index'])
output_data3 = interpreter.get_tensor(output_details[2]['index'])
print(f'score map output shape: {output_data1.shape}')
print(f'geo map output shape: {output_data2.shape}')
print(f'geo angle map output shape: {output_data3.shape}')
print(f'score map output mean: {np.mean(output_data1)}')
print(f'geo map output mean: {np.mean(output_data2)}')
print(f'geo angle map output mean: {np.mean(output_data3)}')
