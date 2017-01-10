from __future__ import print_function
from __future__ import absolute_import
import matplotlib.pyplot as plt
from keras.layers import Input
from DataGeneration import BoundingShape, ItoGenerator, print_process, print_drift, create_color_map
from ManifoldLearningDnn import one_level, get_patches
from ObservationModes import *
import numpy.matlib
import numpy
import os.path
import time
import downhill
import os
import sys
import timeit
import theano.tensor as T
import theano.tensor.nlinalg
import scipy
from sklearn.decomposition import PCA
from Autoencoder import AutoEncoder
from sklearn.manifold import Isomap
import Util
from Util import *
import numpy
import DataGeneration
from DataGeneration import *
import pickle


sim_dir_name = "2D Ring Potential"

intrinsic_test_filename = 'intrinsic_base.txt'
sensor_noisy_test_filename = 'sensor_noisy_base.txt'

sim_dir = './' + sim_dir_name

measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy')

intrinsic_process_test = numpy.loadtxt(sim_dir + '/' + intrinsic_test_filename, delimiter=',').T
sensor_noisy_test = numpy.loadtxt(sim_dir + '/' + sensor_noisy_test_filename, delimiter=',').T
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy')
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy')

f = open(sim_dir + '/' + 'net.save', 'rb')
ca = pickle.load(f)
f.close()

input_base_Theano = T.dmatrix('input_base_Theano')
input_step_Theano = T.dmatrix('input_step_Theano')
measurements_base_Theano = T.dmatrix('measurements_base_Theano')
initial_guess_base_Theano = T.dmatrix('initial_guess_base_Theano')
dim_intrinsic = intrinsic_process_test.shape[0]

rec_output_base_Theano, gen_output_base_init_Theano, gen_output_base_Theano, inv_output_base_init_Theano, inv_output_base_Theano, drift_inv_Theano, cost_rec_pretrain_Theano, cost_gen_pretrain_Theano, cost_gen_Theano, cost_inv_pretrain_Theano, cost_drift_Theano, cost_inv_Theano, cost_intrinsic_Theano, cost_total_Theano = ca.get_cost(dim_intrinsic, intrinsic_variance, measurement_variance, input_base_Theano, input_step_Theano, measurements_base_Theano, initial_guess_base_Theano)

test_model = theano.function(
    inputs=[input_base_Theano],
    outputs=[rec_output_base_Theano, gen_output_base_Theano, inv_output_base_Theano, drift_inv_Theano],
)


[rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(sensor_noisy_test)
colormap = create_color_map(intrinsic_process_test)

print_process(intrinsic_process_test, color_map=colormap, titleStr='Intrinsic Process')

print_process(sensor_noisy_test.T, color_map=colormap, titleStr='Sensor Noisy')

print_process(inv_output_base_val, color_map=colormap, titleStr='Reconstructed')



