# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
from data_generation import print_process, create_color_map
import numpy
import matplotlib.pyplot as plt

sim_dir_name = "2D Unit Circle - Dynamic - Fishbowl"
process_mode = "Dynamic"

#sim_dir_name = "2D Room - Exact Limits - More Points - No Override"
intrinsic_process_file_name = 'intrinsic_process.npy'

sim_dir = './' + sim_dir_name
intrinsic_process_file = sim_dir + '/' + intrinsic_process_file_name

intrinsic_simulated_process = numpy.load(sim_dir + '/' + intrinsic_process_file_name).astype(dtype=numpy.float64).T
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=numpy.float64)
dist_potential = numpy.load(sim_dir + '/' + 'dist_potential.npy').astype(dtype=numpy.float64)

n_points_used = 1e10

n_points = intrinsic_simulated_process.shape[1]

if process_mode == "Static":
    n_points_used = min(n_points, n_points_used)
    points_used_index = numpy.random.choice(intrinsic_simulated_process.shape[1], size=n_points_used, replace=False)

else:
    n_points_used = min(n_points - 1, n_points_used)
    points_used_index = numpy.random.choice(intrinsic_simulated_process.shape[1]-1, size=n_points_used, replace=False)

intrinsic_points_to_use = intrinsic_simulated_process[:, points_used_index]
dist_potential_to_use = dist_potential[points_used_index]

dim_intrinsic = intrinsic_simulated_process.shape[0]

if process_mode == "Static":

    intrinsic_process_to_measure = numpy.zeros((dim_intrinsic, n_points_used*(dim_intrinsic+1)))

    intrinsic_process_to_measure = intrinsic_process_to_measure.T
    intrinsic_points_to_use = intrinsic_points_to_use.T

    for i_point in range(n_points_used):
        intrinsic_process_to_measure[i_point*(dim_intrinsic+1)+0, :] = intrinsic_points_to_use[i_point, :]
        for i_dim in range(dim_intrinsic):
            intrinsic_process_to_measure[i_point * (dim_intrinsic + 1) + i_dim + 1, :] = intrinsic_points_to_use[i_point, :]
            intrinsic_process_to_measure[i_point * (dim_intrinsic + 1) + i_dim + 1, i_dim] = intrinsic_process_to_measure[i_point * (dim_intrinsic + 1) + i_dim + 1, i_dim] + numpy.sqrt(intrinsic_variance)

    intrinsic_process_to_measure = intrinsic_process_to_measure.T
    intrinsic_points_to_use = intrinsic_points_to_use.T

    print_process(intrinsic_process_to_measure, titleStr="Intrinsic Process to Measure")

else:

    intrinsic_process_base = intrinsic_simulated_process[:, points_used_index]
    intrinsic_process_step = intrinsic_simulated_process[:, points_used_index+1]

    intrinsic_process_to_measure = numpy.zeros((dim_intrinsic, 2*n_points_used))

    intrinsic_process_to_measure = intrinsic_process_to_measure.T
    intrinsic_process_base = intrinsic_process_base.T
    intrinsic_process_step = intrinsic_process_step.T
    for i_point in range(n_points_used):
        intrinsic_process_to_measure[i_point*2 + 0, :] = intrinsic_process_base[i_point, :]
        intrinsic_process_to_measure[i_point*2 + 1, :] = intrinsic_process_step[i_point, :]
    intrinsic_process_to_measure = intrinsic_process_to_measure.T
    intrinsic_process_base = intrinsic_process_base.T
    intrinsic_process_step = intrinsic_process_step.T

    print_process(intrinsic_process_base, titleStr="Intrinsic Base Process")
    print_process(intrinsic_process_step, titleStr="Intrinsic Step Process")

numpy.savetxt(sim_dir + '/' + 'dist_potential_used.txt', dist_potential_to_use, delimiter=',')
numpy.savetxt(sim_dir + '/' + 'intrinsic_used.txt', intrinsic_points_to_use.T, delimiter=',')
numpy.savetxt(sim_dir + '/' + 'intrinsic_process_to_measure.txt', intrinsic_process_to_measure.T, delimiter=',')

plt.show(block=True)
