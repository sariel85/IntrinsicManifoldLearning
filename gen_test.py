# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import BoundingShape, ItoGenerator, print_process, create_color_map
import numpy
import os
from ObservationModes import *

sim_dir_name = "2D Room - Exact Limits - More Points"
intrinsic_test_file_name = 'intrinsic_test.txt'
sim_dir = './' + sim_dir_name

origin = numpy.asarray([-0.5, 4.5])
r_start = 1
r_end = 7
r_speed = 1
theta_total = 12
n_points = 200
# Noiseless Measurement
r_range = numpy.arange(r_start, r_end, (r_end-r_start)/n_points)
theta_range = numpy.arange(0, theta_total, (theta_total)/n_points)

x = numpy.multiply(r_range, numpy.cos(theta_range))+origin[0]
y = numpy.multiply(r_range, numpy.sin(theta_range))+origin[1]
diff_x = x[1:]-x[0:-1]
diff_y = y[1:]-y[0:-1]
theta_view = numpy.arctan(diff_y / diff_x)
theta_view[numpy.where(diff_x < 0)] = theta_view[numpy.where(diff_x < 0)] + numpy.pi
theta_view = numpy.insert(theta_view, 0, 0)
theta_view[0] = theta_view[1]
intrinsic_process_test = numpy.asarray([x, y, theta_view]).T

numpy.savetxt(sim_dir + '/' + intrinsic_test_file_name, intrinsic_process_test, delimiter=',')

print_process(intrinsic_process_test[:, 0:2].T, titleStr="Intrinsic Test Process")
print_process(intrinsic_process_test[:, 0:2].T, titleStr="Intrinsic Test Process")
