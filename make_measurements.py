# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import BoundingShape, ItoGenerator, print_process, create_color_map
import matplotlib.pyplot as plt
from ObservationModes import *

sim_dir_name = "Non Convex"
intrinsic_process_file_name = 'intrinsic_process.npy'
sim_dir = './' + sim_dir_name

measurement_variance = 0.0000005

intrinsic_process_file = sim_dir + '/' + intrinsic_process_file_name

intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',')
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',')

n_points = intrinsic_process_base.shape[1]

# Noiseless Measurement
#exact_sensor_base = twirl(intrinsic_process_base, k=6)
#exact_sensor_step = twirl(intrinsic_process_step, k=6)

'''
ant_1 = numpy.asarray([[10], [-7]])
ant_2 = numpy.asarray([[10], [20]])
ant_3 = numpy.asarray([[-16], [5]])
range_factor = [[30], [30], [30]]
antenas = numpy.concatenate((ant_1, ant_2, ant_3), axis=1)
amplitudes = [[1],[3],[7]]
exact_sensor_base = antena(intrinsic_process_base, centers=antenas, amplitudes=amplitudes, range_factor=range_factor)
exact_sensor_step = antena(intrinsic_process_step, centers=antenas, amplitudes=amplitudes, range_factor=range_factor)
'''
#exact_sensor_base = twirl(intrinsic_process_base, k=0.3)
#exact_sensor_step = twirl(intrinsic_process_step, k=0.3)
#exact_sensor_base = swissroll(intrinsic_process_base, k=8)
#exact_sensor_step = swissroll(intrinsic_process_step, k=8)

#exact_sensor_base = singers_mushroom(intrinsic_process_base)
#exact_sensor_step = singers_mushroom(intrinsic_process_step)
exact_sensor_base = whole_sphere(intrinsic_process_base)
exact_sensor_step = whole_sphere(intrinsic_process_step)


# Realistic Measurement
noisy_sensor_base = exact_sensor_base + numpy.sqrt(measurement_variance) * numpy.random.randn(exact_sensor_base.shape[0], n_points)
noisy_sensor_step = exact_sensor_step + numpy.sqrt(measurement_variance) * numpy.random.randn(exact_sensor_step.shape[0], n_points)

numpy.savetxt(sim_dir + '/' + 'sensor_clean_base.txt', exact_sensor_base, delimiter=',')
numpy.savetxt(sim_dir + '/' + 'sensor_clean_step.txt', exact_sensor_step, delimiter=',')

numpy.savetxt(sim_dir + '/' + 'sensor_noisy_base.txt', noisy_sensor_base, delimiter=',')
numpy.savetxt(sim_dir + '/' + 'sensor_noisy_step.txt', noisy_sensor_step, delimiter=',')
numpy.save(sim_dir + '/' + 'measurement_variance', measurement_variance)

n_plot_points = 10000
n_plot_points = min(n_points, n_plot_points)
points_plot_index = numpy.random.choice(n_points, size=n_plot_points, replace=False)

color_map = create_color_map(intrinsic_process_base)

print_process(intrinsic_process_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Base Process")
print_process(intrinsic_process_step, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Step Process")
print_process(exact_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Sensor Clean Base Process")
print_process(exact_sensor_step, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Sensor Clean Step Process")
print_process(noisy_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Sensor Noisy Base Process")
print_process(noisy_sensor_step, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Sensor Noisy Step Process")
plt.show(block=True)
