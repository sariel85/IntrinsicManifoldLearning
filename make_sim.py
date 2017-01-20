# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
import matplotlib.pyplot as plt
from DataGeneration import BoundingShape, ItoGenerator, print_process, create_color_map, bounding_potential
from Util import  print_potential
import numpy
import os
import pickle

#intrinsic_variance = 0.001
'''
sim_dir_name = "2D Ring Potential"
k = 0.005
alpha = 0.005
beta = 20
intrinsic_variance = 0.002
def potential_func (point): return ((1/2)*k*numpy.power(numpy.linalg.norm(point), 4)+alpha*numpy.exp(-beta*(numpy.linalg.norm(point))))
print_potential(potential_func, x_low=-1, x_high=1, y_low=-1, y_high=1, step=0.05)
'''
'''
intrinsic_variance = 0.0005
sim_dir_name = "2D Unit Circle"
bounding_shape = BoundingShape(predef_type="2D Unit Circle")
'''
'''
sim_dir_name = "2D Double Gaussian Potential"
intrinsic_variance = 0.002
p1 = (-0.5, 0)
p2 = (0.5, 0)
alpha = 0.002
beta = 2.5
def potential_func (point): return -alpha*(numpy.exp(-beta*numpy.square(numpy.linalg.norm(point.T-numpy.asarray(p1).T)))+numpy.exp(-beta*numpy.square(numpy.linalg.norm(point.T-numpy.asarray(p2).T))))
'''

#sim_dir_name = "2D Triple Gaussian Potential"
#p1 = (-0.5, 0-0.4)
#p2 = (0.5, 0-0.4)
#p3 = (0, 0.866-0.4)
#alpha = 0.0002
#beta = 2.5
#def potential_func (point): return -alpha*(numpy.exp(-beta*numpy.square(numpy.linalg.norm(point.T-numpy.asarray(p1).T)))+numpy.exp(-beta*numpy.square(numpy.linalg.norm(point.T-numpy.asarray(p2).T)))+numpy.exp(-beta*numpy.square(numpy.linalg.norm(point.T-numpy.asarray(p3).T))))

#sim_dir_name = "2D Triple Gaussian "
'''
sim_dir_name = "2D Room - Exact Limits - More Points"
intrinsic_variance = 0.1

bounding_shape = BoundingShape( vertices=[(7.7, 15.6), (7.7, -3.96), (-6.3, -3.96), (-6.3, -2.54), (-9.30, -2.54), (-9.30, 8.4), (-11.15, 8.4),
              (-11.15, 11.95), (-9.35, 11.95), (-9.35, 10.85), (0.05, 10.85), (0.05, 15.6)])


def potential_func (point):
    bounding_shape = BoundingShape(vertices=[(7.7, 15.6), (7.7, -3.96), (-6.3, -3.96), (-6.3, -2.54), (-9.30, -2.54), (-9.30, 8.4), (-11.15, 8.4), (-11.15, 11.95), (-9.35, 11.95), (-9.35, 10.85), (0.05, 10.85), (0.05, 15.6)])
    return bounding_potential(point, bounding_shape)
    print_potential(potential_func, x_low=-18, x_high=18, y_low=-13, y_high=25, step=0.2)

'''
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
intrinsic_variance = 0.0001
sim_dir_name = "2D Unit Square"
bounding_shape = BoundingShape(predef_type="2D Unit Square")
#print_potential(potential_func, x_low=-1, x_high=1, y_low=-1, y_high=1, step=0.05)
'''
'''
intrinsic_variance = 0.0001
sim_dir_name = "2D Unit Circle"
bounding_shape = BoundingShape(predef_type="2D Unit Circle")
#print_potential(potential_func, x_low=-1, x_high=1, y_low=-1, y_high=1, step=0.05)
'''

'''
intrinsic_variance = 0.001
sim_dir_name = "Rectangle"
bounding_shape = BoundingShape(predef_type="2D Unit Square")
#print_potential(potential_func, x_low=-1, x_high=1, y_low=-1, y_high=1, step=0.05)
'''



sim_dir_name = "Non Convex"
intrinsic_variance = 0.001

bounding_shape = BoundingShape( vertices=[(0,0), (1, 0), (1, 0.33), (0.33, 0.33), (0.33, 0.66), (1, 0.66), (1, 1), (0, 1)])


def potential_func (point):
    bounding_shape = BoundingShape(vertices=[(7.7, 15.6), (7.7, -3.96), (-6.3, -3.96), (-6.3, -2.54), (-9.30, -2.54), (-9.30, 8.4), (-11.15, 8.4), (-11.15, 11.95), (-9.35, 11.95), (-9.35, 10.85), (0.05, 10.85), (0.05, 15.6)])
    return bounding_potential(point, bounding_shape)
    print_potential(potential_func, x_low=-18, x_high=18, y_low=-13, y_high=25, step=0.2)


'''
sim_dir_name = "Triangle"
intrinsic_variance = 0.001

bounding_shape = BoundingShape( vertices=[(0,0), (0.5, 1), (1, 1)])


def potential_func (point):
    bounding_shape = BoundingShape(vertices=[(7.7, 15.6), (7.7, -3.96), (-6.3, -3.96), (-6.3, -2.54), (-9.30, -2.54), (-9.30, 8.4), (-11.15, 8.4), (-11.15, 11.95), (-9.35, 11.95), (-9.35, 10.85), (0.05, 10.85), (0.05, 15.6)])
    return bounding_potential(point, bounding_shape)
    print_potential(potential_func, x_low=-18, x_high=18, y_low=-13, y_high=25, step=0.2)
'''



#sim_dir_name = "2D Room - Star Subset"
#sim_dir_name = "2D Triple Parabolic Potential - 001"
intrinsic_process_file_name = 'intrinsic_process'
intrinsic_variance_file_name = 'intrinsic_variance'
sim_dir = './' + sim_dir_name

if not(os.path.isdir(sim_dir)):
    os.makedirs(sim_dir)

intrinsic_process_file = sim_dir + '/' + intrinsic_process_file_name

# Intrinsic process properties
precision = 'float64'
n_points_simulated = 100001
subsample_factor = 10

#bounding_shape = BoundingShape(predef_type='3D Unit Cube')
#bounding_shape = BoundingShape(predef_type='Star', k=5, r_in=3, r_out=7, origin=[-1.5, 2.7])


#ito_generator = ItoGenerator(intrinsic_potential=potential_func, dim_intrinsic=2, bounding_shape=None)

ito_generator = ItoGenerator(bounding_shape=bounding_shape, dim_intrinsic=2, intrinsic_potential=None)

intrinsic_simulated_process = ito_generator.gen_process(n_trajectory_points=n_points_simulated, process_var=intrinsic_variance, subsample_factor=subsample_factor)

numpy.save(sim_dir + '/' + intrinsic_process_file_name, intrinsic_simulated_process)
numpy.save(sim_dir + '/' + intrinsic_variance_file_name, intrinsic_variance)

color_map = create_color_map(intrinsic_simulated_process)
n_plot_points = 3000
n_plot_points = min(n_points_simulated, n_plot_points)
points_plot_index = numpy.random.choice(n_points_simulated, size=n_plot_points, replace=False)
print_process(intrinsic_simulated_process, bounding_shape=None, indexs=points_plot_index, titleStr="Intrinsic Process", color_map=color_map)
plt.show(block=True)

#bounding_shape = BoundingShape(
#    vertices=[(7.7, 15.6), (7.7, -3.96), (-6.3, -3.96), (-6.3, -2.54), (-9.30, -2.54), (-9.30, 8.4), (-11.15, 8.4),
#              (-11.15, 11.95), (-9.35, 11.95), (-9.35, 10.85), (0.05, 10.85), (0.05, 15.6)])
#ax = print_process(intrinsic_simulated_process, bounding_shape=bounding_shape, titleStr="Points Visited by Agent's Random Walk")
#plt.quiver(intrinsic_simulated_process[0, :-1].T, intrinsic_simulated_process[1, :-1].T, intrinsic_simulated_process[0, 1:] - intrinsic_simulated_process[0, :-1].T, intrinsic_simulated_process[1, 1:] - intrinsic_simulated_process[1, :-1].T, units='xy', scale=1, pivot='tail', width=0.05)


