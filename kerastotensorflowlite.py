import os
import argparse
import numpy as np
import tensorflow as tf
from freeze import freeze_session
from keras.models import model_from_json
from keras import backend as K
K.set_learning_phase(0)

parser = argparse.ArgumentParser(
    description='Converts Keras Model to Tensorflow Model'
)
parser.add_argument(
    '-j', '--json_keras',
    required=True,
    help='Keras json model'
)
parser.add_argument(
    '-k', '--keras',
    required=True,
    help='Keras model'
)
parser.add_argument(
    '-n', '--name',
    required=True,
    help='Model Name'
)
args = parser.parse_args()

model = model_from_json(open(args.json_keras).read(), custom_objects={'tf': tf, 'RESIZE_FACTOR': 2})
model.load_weights(args.keras)

input_arrays = ['input_image']

output_names = [out.op.name for out in model.outputs]

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])


tf.train.write_graph(frozen_graph, args.name, 'frozen_inference_graph.pb', as_text=False)
tf_model_path = os.path.join(args.name, 'frozen_inference_graph.pb')


converter = tf.lite.TFLiteConverter.from_frozen_graph(tf_model_path, input_arrays, output_names,
                                                      input_shapes={'input_image': [1, 512, 512, 3]})
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
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data1 = interpreter.get_tensor(output_details[0]['index'])
output_data2 = interpreter.get_tensor(output_details[1]['index'])
print(output_data1.shape)
print(output_data2.shape)