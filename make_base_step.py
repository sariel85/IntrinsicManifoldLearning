# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import print_process, create_color_map
import numpy
sim_dir_name = "2D Unit Circle"

#sim_dir_name = "2D Room - Exact Limits - More Points - No Override"
intrinsic_process_file_name = 'intrinsic_process.npy'
sim_dir = './' + sim_dir_name

intrinsic_process_file = sim_dir + '/' + intrinsic_process_file_name

n_points_used = 10000

intrinsic_simulated_process = numpy.load(sim_dir + '/' + intrinsic_process_file_name)

n_points = intrinsic_simulated_process.shape[1]

n_points_used = min(n_points-1, n_points_used)

points_used_index = numpy.random.choice(intrinsic_simulated_process.shape[1]-1, size=n_points_used, replace=False)

intrinsic_process_base = intrinsic_simulated_process[:, points_used_index]
intrinsic_process_step = intrinsic_simulated_process[:, points_used_index+1]

numpy.savetxt(sim_dir + '/' + 'intrinsic_base.txt', intrinsic_process_base, delimiter=',')
numpy.savetxt(sim_dir + '/' + 'intrinsic_step.txt', intrinsic_process_step, delimiter=',')


print_process(intrinsic_process_base, titleStr="Intrinsic Base Process")
print_process(intrinsic_process_step, titleStr="Intrinsic Step Process")
'''
intrinsic_process_base_2 = intrinsic_simulated_process[:, points_used_index[50000:]]
intrinsic_process_step_2 = intrinsic_simulated_process[:, points_used_index[50000:]+1]

numpy.savetxt(sim_dir + '/' + 'intrinsic_base_2.txt', intrinsic_process_base_2.T, delimiter=',')
numpy.savetxt(sim_dir + '/' + 'intrinsic_step_2.txt', intrinsic_process_step_2.T, delimiter=',')

print_process(intrinsic_process_base, titleStr="Intrinsic Base Process")
print_process(intrinsic_process_step, titleStr="Intrinsic Step Process")
'''



