from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import print_process, print_drift, create_color_map
import time
import downhill
import os
import theano.tensor as T
import theano.tensor.nlinalg
from Autoencoder import AutoEncoder
from sklearn.manifold import Isomap
from Util import *
import pickle
from sklearn import preprocessing
from sklearn import manifold
from matplotlib import rc
import numpy
from diff_maps import diff_maps

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#T.config.profile = True


sim_dir_name = "Non Convex"
sim_dir = './' + sim_dir_name

n_cluster_points = 2000

#Initial Guess Properties
n_neighbors_cov = 30
n_neighbors_mds = 10

#Visualization properties
n_plot_points = 2000
precision = 'float64'

if precision == 'float32':
    dtype = numpy.float32

elif precision == 'float64':
    dtype = numpy.float64

#############################################################################################


#Create unique folder based on timestamp to save figures in
ts = time.time()
ts = round(ts)
dir_name = '/{}'.format(ts)
full_dir_name = './' + 'Runs' + '/' + sim_dir_name + dir_name
os.makedirs(full_dir_name)

intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',', dtype=dtype).T
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',', dtype=dtype).T
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy')
intrinsic_variance = intrinsic_variance.astype(dtype=dtype)

dim_intrinsic = intrinsic_process_base.shape[0]

n_points = intrinsic_process_base.shape[1]

noisy_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_base.txt', delimiter=',', dtype=dtype)
noisy_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_step.txt', delimiter=',', dtype=dtype)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

#############################################################################################

n_plot_points = min(n_points, n_plot_points)
points_plot_index = numpy.random.choice(n_points, size=n_plot_points, replace=False)

#############################################################################################

# Generate point coloring based on intrinsic coordinates
color_map = create_color_map(intrinsic_process_base)

#############################################################################################

# Get Low Dimensional Starting Point
n_cluster_points = min(n_points, n_cluster_points)
points_cluster_index = numpy.random.choice(n_points, size=n_cluster_points, replace=False)
noisy_sensor_base_for_cluster = noisy_sensor_base[:, points_cluster_index]
intrinsic_process_base_for_cluster = intrinsic_process_base[:, points_cluster_index]

ax_inv = print_process(intrinsic_process_base, indexs=points_cluster_index, bounding_shape=None, color_map=color_map,
                       titleStr="Intrinsic Space")
ax_inv.set_xlabel("$\displaystyle x_1$")
ax_inv.set_ylabel("$\displaystyle x_2$")
plt.savefig(full_dir_name + '/' + 'intrinsic_base.png', bbox_inches='tight')
# plt.close()
# print_process(exact_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Noiseless Sensor Process")
ax_gen = print_process(noisy_sensor_base, indexs=points_cluster_index, bounding_shape=None, color_map=color_map,
                       titleStr="Measurement Space")
ax_gen.set_xlabel("$\displaystyle y_1$")
ax_gen.set_ylabel("$\displaystyle y_2$")
plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')
plt.axis([-0.2, 2, -1, 1.2])

# plt.close()

xlim_inv = ax_inv.get_xlim()
ylim_inv = ax_inv.get_ylim()

if intrinsic_process_base.shape[0] == 2:
    ax_limits_inv = [xlim_inv[0], xlim_inv[1], ylim_inv[0], ylim_inv[1]]
elif intrinsic_process_base.shape[0] == 3:
    zlim_inv = ax_inv.get_zlim()
    ax_limits_inv = [xlim_inv[0], xlim_inv[1], ylim_inv[0], ylim_inv[1], zlim_inv[0], zlim_inv[1]]
else:
    ax_limits_inv = None

if noisy_sensor_base.shape[0] == 2:
    xlim_gen = ax_gen.get_xlim()
    ylim_gen = ax_gen.get_ylim()
    ax_limits_gen = [xlim_gen[0], xlim_gen[1], ylim_gen[0], ylim_gen[1]]
elif noisy_sensor_base.shape[0] == 3:
    xlim_gen = ax_gen.get_xlim()
    ylim_gen = ax_gen.get_ylim()
    zlim_gen = ax_gen.get_zlim()
    ax_limits_gen = [xlim_gen[0], xlim_gen[1], ylim_gen[0], ylim_gen[1], zlim_gen[0], zlim_gen[1]]
else:
    ax_limits_gen = None3


ax_inv = print_process(intrinsic_process_base, indexs=points_cluster_index, bounding_shape=None, color_map=color_map,
                       titleStr="Intrinsic Space")
ax_inv.set_xlabel("$\displaystyle x_1$")
ax_inv.set_ylabel("$\displaystyle x_2$")
plt.savefig(full_dir_name + '/' + 'intrinsic_base.png', bbox_inches='tight')
# plt.close()
# print_process(exact_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Noiseless Sensor Process")
ax_gen = print_process(noisy_sensor_base, indexs=points_cluster_index, bounding_shape=None, color_map=color_map,
                       titleStr="Measurement Space")
ax_gen.set_xlabel("$\displaystyle y_1$")
ax_gen.set_ylabel("$\displaystyle y_2$")
plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')
plt.axis([-0.2, 2, -1, 1.2])

# plt.close()
# print_process(exact_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Noiseless Sensor Process")
ax_gen = print_process(noisy_sensor_base, indexs=points_cluster_index[:10], bounding_shape=None, color_map=color_map,
                       titleStr="Measurement Space", ax=ax_gen)
ax_gen.set_xlabel("$\displaystyle y_1$")
ax_gen.set_ylabel("$\displaystyle y_2$")
plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')
plt.axis([-0.2, 2, -1, 1.2])

pos_1, pos_2, pos_3, diff_embedding, diff_embedding_non_int = diff_maps(points_cluster_index=points_cluster_index, noisy_sensor_base=noisy_sensor_base, noisy_sensor_step=noisy_sensor_step, intrinsic_variance=intrinsic_variance, intrinsic_process_base=intrinsic_process_base, intrinsic_process_step=intrinsic_process_step, dim_intrinsic=2, n_neighbors_cov = 200, n_neighbors_mds = 20, ax_limits_inv=ax_limits_inv, ax_limits_gen=ax_limits_gen)

print_process(pos_1.T, bounding_shape=None, color_map=color_map[points_cluster_index, :], titleStr="Suggested Intrinsic Isomap", align_points=intrinsic_process_base_for_cluster)
stress, stress_normlized = embbeding_score(intrinsic_process_base_for_cluster, pos_1.T, titleStr="Suggested Intrinsic Isomap")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

print_process(pos_2.T, bounding_shape=None, color_map=color_map[points_cluster_index, :], titleStr="Isomaps using True Intrinsic Distances", align_points=intrinsic_process_base_for_cluster)
stress, stress_normlized = embbeding_score(intrinsic_process_base_for_cluster, pos_2.T, titleStr="Isomaps using True Intrinsic Distances")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

print_process(pos_3.T, bounding_shape=None, color_map=color_map[points_cluster_index, :], titleStr="Isomaps using Measured Distances", align_points=intrinsic_process_base_for_cluster)
stress, stress_normlized = embbeding_score(intrinsic_process_base_for_cluster, pos_3.T, titleStr="Isomaps using Measured Distances")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

print_process(diff_embedding.T, bounding_shape=None, color_map=color_map[points_cluster_index, :], titleStr="Intrinsic Diffusion Maps", align_points=intrinsic_process_base_for_cluster)
stress, stress_normlized = embbeding_score(intrinsic_process_base_for_cluster, 2*6.5*diff_embedding.T, titleStr="Intrinsic Diffusion Maps")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

print_process(diff_embedding_non_int.T, bounding_shape=None, color_map=color_map[points_cluster_index, :], titleStr="Diffusion Maps on Intrinsic Space", align_points=intrinsic_process_base_for_cluster)
stress, stress_normlized = embbeding_score(intrinsic_process_base_for_cluster, 2*6.5*diff_embedding.T, titleStr="Diffusion Maps on Intrinsic Space")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

'''
R, t = rigid_transform(diff_embedding.T, intrinsic_process_base_for_cluster)
diff_embedding_aligned = (numpy.dot(R, diff_embedding.T).T + t.T).T
print_process(diff_embedding_aligned, bounding_shape=None, color_map=color_map[points_cluster_index, :], titleStr="Intrinsic Diffusion Maps after Eigenfunction Inversion", align_points=intrinsic_process_base_for_cluster)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
diff_embedding_aligned = min_max_scaler.fit_transform(diff_embedding_aligned.T).T

print_process(diff_embedding_aligned, bounding_shape=None, color_map=color_map[points_cluster_index, :], titleStr="Intrinsic Diffusion Maps after Eigenfunction Inversion")

diff_embedding_arccos = numpy.arccos(diff_embedding_aligned)/numpy.pi

print_process(diff_embedding_arccos, bounding_shape=None, color_map=color_map[points_cluster_index, :], titleStr="Intrinsic Diffusion Maps after Eigenfunction Inversion")
stress, stress_normlized = embbeding_score(intrinsic_process_base_for_cluster, diff_embedding_arccos, titleStr="Intrinsic Diffusion Maps after Eigenfunction Inversion")
print('stress:', stress)
print('stress_normlized:', stress_normlized)
'''


