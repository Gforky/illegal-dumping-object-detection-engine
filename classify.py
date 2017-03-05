import numpy as np
import tensorflow as tf
import argparse
import sys

FLAGS = None

def create_graph():
    MODEL_PATH = FLAGS.model_dir
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
        graph_def.ParseFromString(f.read())
        del(graph_def.node[1].attr["dct_method"])
        _ = tf.import_graph_def(graph_def, name='')

def run_inference():
    IMAGE_PATH = FLAGS.image_dir
    LABEL_PATH = FLAGS.label_dir
    if not tf.gfile.Exists(IMAGE_PATH):
        tf.logging.fatal('File does not exist %s', IMAGE_PATH)
        return None

    image_data = tf.gfile.FastGFile(IMAGE_PATH, 'rb').read()
    create_graph()

    with tf.Session() as sess:
        final_layer = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(final_layer, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        top_3 = predictions.argsort()[-3:][::-1] 
        
        lines = open(LABEL_PATH, 'rb').readlines()
        labels = [str(line).replace("\n", "") for line in lines]
        
        for category in top_3:
            class_cat = labels[category]
            score = predictions[category]
            print('%s (score = %.4f)' % (class_cat, score))
        
        return labels[top_3[0]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/Users/wendyhu/output_graph.pb',
        help='Path to output_graph.pb'
    )
    parser.add_argument(
        '--label_dir',
        type=str,
        default='/Users/wendyhu/output_labels.txt',
        help='Path to output_labels.txt'
    )
    FLAGS, unparsed = parser.parse_known_args()
    run_inference()

# credit: https://github.com/eldor4do/TensorFlow-Examples/blob/master/retraining-example.py