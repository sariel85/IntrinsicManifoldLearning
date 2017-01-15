from __future__ import print_function
from __future__ import absolute_import
import time
import os
from non_local_tangent import non_local_tangent_net
from DataGeneration import print_process, create_color_map, print_dynamics
from Util import *
from sklearn import manifold
import numpy

###Settings#############################################################################################################
sim_dir_name = "2D Unit Circle" #Which dataset to run

n_points_used_for_dynamics = 800 #How many points are available from which to infer dynamics
n_points_used_for_plotting_dynamics = 500
n_points_used_for_clusters = 200 #How many cluster to use in Kernal method
n_neighbors_cov = 20 #How neighboors to use from which to infer dynamics locally
n_neighbors_mds = 10 #How many short distances are kept for each cluster point
n_hidden_tangent = 10 #How many nodes in hidden layer that learns tangent plane
n_hidden_int = 10 #How many nodes in hidden layer that learns intrinsic dynamics
########################################################################################################################

sim_dir = './' + sim_dir_name
dtype = numpy.float64

intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',', dtype=dtype)
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',', dtype=dtype)
noisy_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_base.txt', delimiter=',', dtype=dtype)
noisy_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_step.txt', delimiter=',', dtype=dtype)
noisy_sensor_base_mean = noisy_sensor_base.mean()
noisy_sensor_base = (noisy_sensor_base - noisy_sensor_base_mean)*1.5
noisy_sensor_step = (noisy_sensor_step - noisy_sensor_base_mean)*1.5
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=dtype)
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy').astype(dtype=dtype)


#ts = time.time()
#ts = round(ts)
#dir_name = '/{}'.format(ts)
#full_dir_name = './' + 'Runs' + '/' + sim_dir_name + dir_name
#os.makedirs(full_dir_name)

dim_intrinsic = intrinsic_process_base.shape[0]
dim_measurement = noisy_sensor_base.shape[0]
n_points = intrinsic_process_base.shape[1]

n_points_used_for_dynamics = min(n_points, n_points_used_for_dynamics)
points_used_dynamics_index = numpy.random.choice(n_points, size=n_points_used_for_dynamics, replace=False)

intrinsic_process_base = intrinsic_process_base[:, points_used_dynamics_index]
intrinsic_process_step = intrinsic_process_step[:, points_used_dynamics_index]
noisy_sensor_base = noisy_sensor_base[:, points_used_dynamics_index]
noisy_sensor_step = noisy_sensor_step[:, points_used_dynamics_index]

n_points = intrinsic_process_base.shape[1]

n_points_used_for_plotting_dynamics = min(n_points, n_points_used_for_plotting_dynamics)
points_dynamics_plot_index = numpy.random.choice(n_points, size=n_points_used_for_plotting_dynamics, replace=False)

color_map = create_color_map(intrinsic_process_base)

print_process(intrinsic_process_base, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Process")
#plt.savefig(full_dir_name + '/' + 'intrinsic_base.png', bbox_inches='tight')

print_process(noisy_sensor_base, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Measurement Process")
#plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')

print_dynamics(intrinsic_process_base, intrinsic_process_step, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Process Dynamics")
#plt.savefig(full_dir_name + '/' + 'intrinsic_dynamics.png', bbox_inches='tight')

print_dynamics(noisy_sensor_base, noisy_sensor_step, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Measurement Process Dynamics")
#plt.savefig(full_dir_name + '/' + 'sensor_dynamics.png', bbox_inches='tight')


###Intrinsic Metric Learning Net########################################################################################
non_local_tangent_net_instance = non_local_tangent_net(intrinsic_process_base, dim_measurements=dim_measurement, dim_intrinsic=dim_intrinsic, n_hidden_tangent=n_hidden_tangent,  n_hidden_int=n_hidden_int, intrinsic_variance=intrinsic_variance, measurement_variance=measurement_variance)
non_local_tangent_net_instance.train_net(noisy_sensor_base, noisy_sensor_step)
########################################################################################################################

#Testing and comparison with other methods##############################################################################
n_points_used_for_clusters = min(n_points, n_points_used_for_clusters)
n_points_used_for_clusters_indexs = numpy.random.choice(n_points, size=n_points_used_for_clusters, replace=False)


#Measured Points
noisy_sensor_clusters = noisy_sensor_base[:, n_points_used_for_clusters_indexs]

#Ground Truth
intrinsic_process_clusters = intrinsic_process_base[:, n_points_used_for_clusters_indexs]

n_points_used_for_clusters = intrinsic_process_clusters.shape[1]


color_map_clusters = color_map[n_points_used_for_clusters_indexs, :]

metric_list_net_tangent, metric_list_net_intrinsic = get_metrics_from_net(non_local_tangent_net_instance, noisy_sensor_clusters)

print_metrics(noisy_sensor_clusters, metric_list_net_tangent, intrinsic_dim=dim_intrinsic, titleStr="Net Learned Tangent Space", scale=intrinsic_variance, space_mode=True)

print_metrics(noisy_sensor_clusters, metric_list_net_intrinsic, intrinsic_dim=dim_intrinsic, titleStr="Net Learned Intrinsic Jacobians", scale=intrinsic_variance, space_mode=False)

metric_list_def, metric_list_full = get_metrics_from_points(noisy_sensor_clusters, noisy_sensor_base, noisy_sensor_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance)

print_metrics(noisy_sensor_clusters, metric_list_def, intrinsic_dim=dim_intrinsic, titleStr="Locally Learned Tangent Jacobians", scale=intrinsic_variance, space_mode=True)

print_metrics(noisy_sensor_clusters, metric_list_def, intrinsic_dim=dim_intrinsic, titleStr="Locally Learned Intrinsic Jacobians", scale=intrinsic_variance, space_mode=False)


dist_mat_net_tangent = numpy.sqrt(calc_dist(noisy_sensor_clusters, metric_list_net_tangent))
dist_mat_net_intrinsic = numpy.sqrt(calc_dist(noisy_sensor_clusters, metric_list_net_intrinsic))
dist_mat_true = numpy.sqrt(calc_dist(intrinsic_process_clusters))
dist_mat_measured = numpy.sqrt(calc_dist(noisy_sensor_clusters))
dist_mat_local = numpy.sqrt(calc_dist(noisy_sensor_clusters, metric_list_def))


dist_mat_net_tangent = trim_distances(dist_mat_net_tangent, dist_mat_measured, n_neighbors=n_neighbors_mds)
dist_mat_net_intrinsic = trim_distances(dist_mat_net_intrinsic, dist_mat_measured, n_neighbors=n_neighbors_mds)
dist_mat_local = trim_distances(dist_mat_local, dist_mat_measured, n_neighbors=n_neighbors_mds)
dist_mat_true_sp = trim_distances(dist_mat_true, n_neighbors=n_neighbors_mds)
dist_mat_measured_sp = trim_distances(dist_mat_measured, n_neighbors=n_neighbors_mds)

dist_mat_net_tangent = scipy.sparse.csgraph.shortest_path(dist_mat_net_tangent, directed=False)
dist_mat_net_intrinsic = scipy.sparse.csgraph.shortest_path(dist_mat_net_intrinsic, directed=False)
dist_mat_local = scipy.sparse.csgraph.shortest_path(dist_mat_local, directed=False)
dist_mat_true_sp = scipy.sparse.csgraph.shortest_path(dist_mat_true_sp, directed=False)
dist_mat_measured_sp = scipy.sparse.csgraph.shortest_path(dist_mat_measured_sp, directed=False)

mds = manifold.MDS(n_components=dim_intrinsic, max_iter=20000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)

iso_embedding_intrinsic = mds.fit(dist_mat_net_intrinsic).embedding_
iso_embedding_tangent = mds.fit(dist_mat_net_tangent).embedding_
iso_embedding_local = mds.fit(dist_mat_local).embedding_
iso_embedding_true_sp = mds.fit(dist_mat_true_sp).embedding_
iso_embedding_measured_sp = mds.fit(dist_mat_measured_sp).embedding_
iso_embedding_true = mds.fit(dist_mat_true).embedding_
iso_embedding_measured = mds.fit(dist_mat_measured).embedding_


diff_embedding_tangent = calc_diff_map(dist_mat_net_tangent, dims=2, factor=2)
diff_embedding_intrinsic = calc_diff_map(dist_mat_net_intrinsic, dims=2, factor=2)
diff_embedding_local = calc_diff_map(dist_mat_local, dims=2, factor=2)
diff_embedding_true_sp = calc_diff_map(dist_mat_true_sp, dims=2, factor=2)
diff_embedding_measured_sp = calc_diff_map(dist_mat_true_sp, dims=2, factor=2)
diff_embedding_true = calc_diff_map(dist_mat_local, dims=2, factor=2)
diff_embedding_measured = calc_diff_map(dist_mat_true_sp, dims=2, factor=2)


###Isomaps##############################################################################################################
print_process(iso_embedding_intrinsic.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Isomap with Net-Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_intrinsic.T, titleStr="Isomap with Net-Learned Intrinsic Metric", n_points=n_points_used_for_clusters)
print('iso_embedding_intrinsic:', stress_normlized)

print_process(iso_embedding_tangent.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Isomap with Net-Learned Tangent Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_tangent.T, titleStr="Isomap with Net-Learned Tangent Metric", n_points=n_points_used_for_clusters)
print('iso_embedding_tangent:', stress_normlized)

print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_local.T, titleStr="Isomap with Locally Learned Intrinsic Metric", n_points=n_points_used_for_clusters)
print('iso_embedding_local:', stress_normlized)

print_process(iso_embedding_true_sp.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Isomap with Exact Short Distances", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_true_sp.T, titleStr="Isomap with Exact Short Distances", n_points=n_points_used_for_clusters)
print('iso_embedding_true_sp:', stress_normlized)

print_process(iso_embedding_measured_sp.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Regular Isomap", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_measured_sp.T, titleStr="Regular Isomap", n_points=n_points_used_for_clusters)
print('iso_embedding_measured_sp:', stress_normlized)

print_process(iso_embedding_true.T, bounding_shape=None, color_map=color_map_clusters, titleStr="PCA on Intrinsic Space", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_true.T, titleStr="PCA on Intrinsic Space", n_points=n_points_used_for_clusters)
print('iso_embedding_true:', stress_normlized)

print_process(iso_embedding_measured.T, bounding_shape=None, color_map=color_map_clusters, titleStr="PCA on Measured Space", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_measured.T, titleStr="PCA on Measured Space", n_points=n_points_used_for_clusters)
print('iso_embedding_measured:', stress_normlized)
########################################################################################################################

###Diffusion Maps#######################################################################################################
print_process(diff_embedding_intrinsic.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps with Net-Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_intrinsic.T, titleStr="Diffusion Maps with Net-Learned Intrinsic Metric", n_points=n_points_used_for_clusters)
print('diff_embedding_intrinsic:', stress_normlized)

print_process(diff_embedding_tangent.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps with Net-Learned Tangent Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_tangent.T, titleStr="Diffusion Maps with Net-Learned Tangent Metric", n_points=n_points_used_for_clusters)
print('diff_embedding_tangent:', stress_normlized)

print_process(diff_embedding_local.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_local.T, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric", n_points=n_points_used_for_clusters)
print('diff_embedding_local:', stress_normlized)

print_process(diff_embedding_true_sp.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps with Exact Short Distances", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_true_sp.T, titleStr="Diffusion Maps with Exact Short Distances", n_points=n_points_used_for_clusters)
print('diff_embedding_true_sp:', stress_normlized)

print_process(diff_embedding_true.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps on Intrinsic Space", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_true.T, titleStr="Diffusion Maps on Intrinsic Space", n_points=n_points_used_for_clusters)
print('diff_embedding_true:', stress_normlized)

print_process(diff_embedding_measured.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps on Measured Space", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_measured.T, titleStr="Diffusion Maps on Measured Space", n_points=n_points_used_for_clusters)
print('diff_embedding_measured:', stress_normlized)
########################################################################################################################

plt.show(block=True)
