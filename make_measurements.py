# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
from data_generation import BoundingShape, ItoGenerator, print_process, create_color_map
import matplotlib.pyplot as plt
from observation_modes import *

sim_dir_name = "2D Unit Square - Dynamic"
intrinsic_process_file_name = 'intrinsic_process.npy'
sim_dir = './' + sim_dir_name
intrinsic_process_file = sim_dir + '/' + intrinsic_process_file_name
intrinsic_to_measure = numpy.loadtxt(sim_dir + '/' + 'intrinsic_process_to_measure.txt', delimiter=',').T
n_points = intrinsic_to_measure.shape[1]
n_plot_points = 1200


#sim_dir_name = "2D Unit circle - Static - Fishbowl"
#exact_sensor = whole_sphere(intrinsic_to_measure, k=5)
#measurement_variance = 0.00001


#sim_dir_name = "Non Convex"
exact_sensor = swissroll(intrinsic_to_measure, k=0.)
measurement_variance = 0.00**2



# Noiseless Measurement
#exact_sensor_base = twirl(intrinsic_process_base, k=6)
#exact_sensor_step = twirl(intrinsic_process_step, k=6)


#ant_1 = numpy.asarray([[0.75], [-0.5]])
#ant_2 = numpy.asarray([[1.5], [1.5]])
#ant_3 = numpy.asarray([[-0.5], [1.5]])


#measurement_variance = 0.
#exact_sensor = whole_sphere((intrinsic_to_measure-6)/5)/2
'''
'''
'''
measurement_variance = 0.
ant_1 = numpy.asarray([[0.], [0.]])
ant_2 = numpy.asarray([[0.], [8.]])
ant_3 = numpy.asarray([[5.], [7]])

range_factor = [[2], [2], [2]]
antenas = numpy.concatenate((ant_1, ant_2, ant_3), axis=1)
amplitudes = [[2],[2],[2]]
exact_sensor = antena(intrinsic_to_measure, centers=antenas, amplitudes=amplitudes, range_factor=range_factor)
'''
'''
'''


'''
measurement_variance = 0.
ant_1 = numpy.asarray([[0.75], [0.75]])
ant_2 = numpy.asarray([[-0.25], [0.75]])
ant_3 = numpy.asarray([[0.75], [-0.25]])

range_factor = [[1], [1], [1]]
antenas = numpy.concatenate((ant_1, ant_2, ant_3), axis=1)
amplitudes = [[2],[2],[2]]
exact_sensor = antena(intrinsic_to_measure, centers=antenas, amplitudes=amplitudes, range_factor=range_factor)
'''''

#exact_sensor_base = twirl(intrinsic_process_base, k=0.3)
#exact_sensor_step = twirl(intrinsic_process_step, k=0.3)
#exact_sensor_base = swissroll(intrinsic_process_base, k=8)
#exact_sensor_step = swissroll(intrinsic_process_step, k=8)

'''
measurement_variance = 0.
#exact_sensor = singers_mushroom(intrinsic_to_measure)
exact_sensor = whole_sphere((intrinsic_to_measure-6)/5)/2
'''

# Realistic Measurement
noisy_sensor = exact_sensor + numpy.sqrt(measurement_variance) * numpy.random.randn(exact_sensor.shape[0], n_points)

numpy.savetxt(sim_dir + '/' + 'sensor_clean.txt', exact_sensor.T, delimiter=',')

numpy.savetxt(sim_dir + '/' + 'sensor_noisy.txt', noisy_sensor.T, delimiter=',')

numpy.save(sim_dir + '/' + 'measurement_variance', measurement_variance)

n_plot_points = min(n_points, n_plot_points)
points_plot_index = numpy.random.choice(n_points, size=n_plot_points, replace=False)

color_map = create_color_map(intrinsic_to_measure)

print_process(intrinsic_to_measure, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Process")
print_process(exact_sensor, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Process")
print_process(noisy_sensor, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Process + Noise")

plt.show(block=True)
