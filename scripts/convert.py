"""Save mode """
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import tensorflow as tf
import numpy as np
import align.detect_face
from tensorflow.python.summary import summary

output_node_names = ['pnet/conv4-2/BiasAdd', 'pnet/prob1', 'rnet/conv5-2/conv5-2', 'rnet/prob1', 'onet/conv6-2/conv6-2', 'onet/conv6-3/conv6-3', 'onet/prob1']

def main(args):
    
    with tf.Graph().as_default():
        config = tf.ConfigProto( device_count = {'GPU': 0})
        sess = tf.Session(config = config)
        with sess.as_default():
            print('Creating networks and loading parameters')
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            print('Freezing graph')
            frozen_graph_def = tf.graph_util.convert_variables_to_constants( sess, sess.graph_def, output_node_names)

            # Save the frozen graph
            print("Saving model to ", args.output)
            with open(args.output, 'wb') as f:
                f.write(frozen_graph_def.SerializeToString())

        pb_visual_writer = summary.FileWriter("./log")
        pb_visual_writer.add_graph(sess.graph)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', 
        help='model path', default="mtcnn.pb")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
