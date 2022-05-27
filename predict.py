import argparse
import tensorflow as tf
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',
                    required=True,
                    help='Path of the model file')
parser.add_argument('--image_file',
                    required=True,
                    help='Path of the input image')
args = parser.parse_args()
# python predict.py --model_path /home/code/codelab/model_r_all --image_file /home/code/codelab/dataset_all/test/incorrect/000033_3_000033_NONE_20.jpg

def main():
    # Enable eager execution to evaluate the tensors immediately once an
    # operator is defined
    # tf.enable_eager_execution()
    tf.executing_eagerly()
    # tf.keras.backend.set_session(tf.Session())
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
    # custom_objects={"tf": tf} is used because the Keras model contains
    # Lambda layers using method in tf package.
    model = tf.keras.models.load_model(args.model_path, custom_objects={"tf": tf})
    image = cv2.imread(args.image_file)
    # OpenCV reads image in BGR channel order. However tf.image.decode_image
    # used in training loads image in RGB channel order. Therefore the channels
    # need to be revsersed here.
    image = image[:, :, ::-1]
    # Resize the image to 150x150 in case it is not in the expected shape
    image = cv2.resize(image, (150, 150))
    # expand_dims is applied because the model expects a 4 dimensional input
    # tensor.
    label, prob = model.predict(np.expand_dims(image, 0))
    print(('Predicted labels: %d' % label[0]))
    print(('Probability: %f' % prob[0]))


if __name__ == '__main__':
    main()
