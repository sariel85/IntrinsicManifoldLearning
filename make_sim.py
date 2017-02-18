# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
import matplotlib.pyplot as plt
from data_generation import BoundingShape, ItoGenerator, print_process, create_color_map, bounding_potential
from util import print_potential
import numpy
import os
import pickle

intrinsic_process_file_name = 'intrinsic_process'
intrinsic_variance_file_name = 'intrinsic_variance'
dist_potential_file_name = 'dist_potential'


added_dim_limits = None
n_points_simulated = 100000

precision = 'float64'
subsample_factor = 10

n_plot_points = 10000
boundary_threshold = 0.2

'''
sim_dir_name = "2D Small Room - Static - Camera"
process_mode = "Static"
intrinsic_variance = 0.1**2
bounding_shape = BoundingShape(vertices=[(2.4, 0), (2.4, 1.13), (1, 1.13), (1, 3.15), (0, 3.15), (0, 6), (3, 6), (3, 4.2), (4.2, 4.2), (4.2, 6), (7.5, 6), (7.5, 0), (7.5, 0)])
boundary_threshold = 0.2
'''

'''
sim_dir_name = "2D Non Convex Room - Static - Camera - 1"
process_mode = "Static"
intrinsic_variance = 0.1**2
bounding_shape = BoundingShape(vertices=[(2.4, 0), (2.4, 1.13), (1, 1.13), (1, 3.15), (0, 3.15), (0, 6), (3, 6), (3, 4.2), (4.2, 4.2), (4.2, 6), (15, 6), (15, 4.1), (14.1, 4.1), (14.1, 2), (15, 2), (15, 0), (10.4, 0), (10.4, 3), (7.5, 3), (7.5, 0)])
boundary_threshold = 0.2
'''

'''
sim_dir_name = "3D Small Room"
process_mode = "Static"
intrinsic_variance = 0.1**2
bounding_shape = BoundingShape(vertices=[(2.4, 0), (2.4, 1.13), (1, 1.13), (1, 3.15), (0, 3.15), (0, 6), (3, 6), (3, 4.2), (4.2, 4.2), (4.2, 6), (7.5, 6), (7.5, 0), (7.5, 0)])
added_dim_limits = numpy.asarray([[0., 6.]]).T
boundary_threshold = 0.2
'''

'''
sim_dir_name = "3D Non Convex Room"
process_mode = "Static"
intrinsic_variance = 0.1**2
bounding_shape = BoundingShape(vertices=[(2.4, 0), (2.4, 1.13), (1, 1.13), (1, 3.15), (0, 3.15), (0, 6), (3, 6), (3, 4.2), (4.2, 4.2), (4.2, 6), (7.5, 6), (7.5, 0), (7.5, 0)])
added_dim_limits = numpy.asarray([[0., 6.]]).T
boundary_threshold = 0.2
'''


#sim_dir_name = "2D Unit Square"
#sim_dir_name = "2D Unit Square - Dynamic"
#process_mode = "Dynamic"
#intrinsic_variance = 0.03**2
#bounding_shape = BoundingShape(vertices=[(0, 0), (1, 0), (1, 1), (0, 1)])
#boundary_threshold = 0.



sim_dir_name = "2D Unit Circle - Dynamic - Fishbowl"
process_mode = "Dynamic"
intrinsic_variance = 0.02**2
n_legs = 24
r = 0.5
bounding_shape = BoundingShape(vertices=[(numpy.cos(2*numpy.pi/n_legs*x)*r, numpy.sin(2*numpy.pi/n_legs*x)*r) for x in range(0, n_legs+1)])
boundary_threshold = 0.03


#sim_dir_name = "Non Convex"
#process_mode = "Static"
#intrinsic_variance = 0.05**2
#bounding_shape = BoundingShape(vertices=[(0, 0), (1, 0), (1, 0.33), (0.33, 0.33), (0.33, 0.66), (1, 0.66), (1, 1), (0, 1)])
#boundary_threshold = 0.03

#Generate Data Set
#sim_dir_name = "Non Convex"
#process_mode = "Static"
#intrinsic_variance = 0.05**2

'''
x_width = 1
y_width = 3
hole_start_x = 0.25
hole_start_y = 1.25
hole_width_x = 0.5
hole_width_y = 1.25

intrinsic_points = numpy.zeros((2, n_points_simulated))
intrinsic_points = intrinsic_points.T
for i_point in range(n_points_simulated):
    while 1:
        test_point = numpy.asarray([numpy.random.rand()*x_width, numpy.random.rand()*y_width])
        if not((test_point[0] > hole_start_x) and (test_point[0] < (hole_start_x + hole_width_x)) and (test_point[1] > hole_start_y) and (test_point[1] < hole_start_y + hole_width_y)):
            break
        #break
    intrinsic_points[i_point, :] = test_point
intrinsic_points = intrinsic_points.T
intrinsic_simulated_process = intrinsic_points
'''
# Not in use
'''
sim_dir_name = "2D Water Molecule"
r = 0.7
deg = 104.45
p1 = (0, 0)
p2 = (r, 0)
p3 = (r*numpy.cos((deg/180)*numpy.pi), r*numpy.sin((deg/180)*numpy.pi))
alpha1 = 0.0002
alpha2 = 0.0002
scale = 0.5
beta = 5
def potential_func (point): return -scale*(alpha1*numpy.exp(-beta*numpy.square(numpy.linalg.norm(point.T-numpy.asarray(p1).T)))+alpha2*numpy.exp(-beta*numpy.square(numpy.linalg.norm(point.T-numpy.asarray(p2).T)))+alpha2*numpy.exp(-beta*numpy.square(numpy.linalg.norm(point.T-numpy.asarray(p3).T))))
'''

'''
sim_dir_name = "Triangle"
intrinsic_variance = 0.001
bounding_shape = BoundingShape( vertices=[(0,0), (0.5, 1), (1, 1)])
def potential_func (point):
    bounding_shape = BoundingShape(vertices=[(7.7, 15.6), (7.7, -3.96), (-6.3, -3.96), (-6.3, -2.54), (-9.30, -2.54), (-9.30, 8.4), (-11.15, 8.4), (-11.15, 11.95), (-9.35, 11.95), (-9.35, 10.85), (0.05, 10.85), (0.05, 15.6)])
    return bounding_potential(point, bounding_shape)
    print_potential(potential_func, x_low=-18, x_high=18, y_low=-13, y_high=25, step=0.2)
'''

'''
sim_dir_name = "2D Ring Potential"
k = 0.005
alpha = 0.005
beta = 20
intrinsic_variance = 0.002
def potential_func (point): return ((1/2)*k*numpy.power(numpy.linalg.norm(point), 4)+alpha*numpy.exp(-beta*(numpy.linalg.norm(point))))
print_potential(potential_func, x_low=-1, x_high=1, y_low=-1, y_high=1, step=0.05)
'''

sim_dir = './' + sim_dir_name

if not(os.path.isdir(sim_dir)):
    os.makedirs(sim_dir)

intrinsic_process_file = sim_dir + '/' + intrinsic_process_file_name


#ito_generator = ItoGenerator(intrinsic_potential=potential_func, dim_intrinsic=2, bounding_shape=None)

ito_generator = ItoGenerator(bounding_shape=bounding_shape, added_dim_limits=added_dim_limits)

intrinsic_simulated_process, dist_potential = ito_generator.gen_process(n_trajectory_points=n_points_simulated, process_var=intrinsic_variance, process_mode=process_mode, added_dim_limits=added_dim_limits, subsample_factor=subsample_factor)

numpy.save(sim_dir + '/' + intrinsic_process_file_name, intrinsic_simulated_process.T)
numpy.save(sim_dir + '/' + intrinsic_variance_file_name, intrinsic_variance)
numpy.save(sim_dir + '/' + dist_potential_file_name, dist_potential)

color_map = create_color_map(intrinsic_simulated_process)
n_plot_points = min(n_points_simulated, n_plot_points)
points_plot_index = numpy.random.choice(n_points_simulated, size=n_plot_points, replace=False)
print_process(intrinsic_simulated_process, bounding_shape=None, indexs=points_plot_index, titleStr="Intrinsic Space", color_map=color_map)
#color_map[numpy.where(dist_potential < boundary_threshold), :] = [0, 0, 0]
#color_map[numpy.where(dist_potential > boundary_threshold), :] = color_map[numpy.where(dist_potential > boundary_threshold), :]
print_process(intrinsic_simulated_process, bounding_shape=None, indexs=points_plot_index, titleStr="Intrinsic Space + Boundaries", color_map=color_map)
plt.show(block=True)

