from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import print_process, create_color_map, print_dynamics
from Util import *
import numpy
from non_local_tangent import non_local_tangent_net
###Settings#############################################################################################################
sim_dir_name = "2D Unit Circle" #Which dataset to run

n_points_used_for_dynamics = 400 #How many points are available from which to infer dynamics
n_points_used_for_plotting_dynamics = 400
n_points_used_for_clusters = 400 #How many cluster to use in Kernal method
n_points_used_for_clusters_2 = 400 #How many cluster to use in Kernal method

n_neighbors_cov = 10 #How neighboors to use from which to infer dynamics locally
n_neighbors_mds = 10 #How many short distances are kept for each cluster point
n_hidden_drift = 4 #How many nodes in hidden layer that learns intrinsic dynamics
n_hidden_tangent = 30 #How many nodes in hidden layer that learns tangent plane
n_hidden_int = 10 #How many nodes in hidden layer that learns intrinsic dynamics
########################################################################################################################

sim_dir = './' + sim_dir_name
dtype = numpy.float64

intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',', dtype=dtype)
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',', dtype=dtype)
noisy_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_base.txt', delimiter=',', dtype=dtype)
noisy_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_step.txt', delimiter=',', dtype=dtype)
noisy_sensor_base_mean = noisy_sensor_base.mean()
noisy_sensor_base = (noisy_sensor_base - noisy_sensor_base_mean)
noisy_sensor_step = (noisy_sensor_step - noisy_sensor_base_mean)
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=dtype)
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy').astype(dtype=dtype)

intrinsic_process_base = intrinsic_process_base[:, :noisy_sensor_base.shape[1]]
intrinsic_process_step = intrinsic_process_step[:, :noisy_sensor_step.shape[1]]


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

print_process(intrinsic_process_base, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Space")
#plt.savefig(full_dir_name + '/' + 'intrinsic_base.png', bbox_inches='tight')

print_process(noisy_sensor_base, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Space")
#plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')

print_dynamics(intrinsic_process_base, intrinsic_process_step, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Process Dynamics")
#plt.savefig(full_dir_name + '/' + 'intrinsic_dynamics.png', bbox_inches='tight')

#print_dynamics(noisy_sensor_base, noisy_sensor_step, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Process Dynamics")
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

#net_drift = get_drift_from_net(non_local_tangent_net_instance, noisy_sensor_clusters)



#print_drift(noisy_sensor_clusters, net_drift, titleStr="Net Learned Drift")

print_metrics(noisy_sensor_clusters, metric_list_net_tangent, intrinsic_dim=dim_intrinsic, titleStr="Net Learned Tangent Space", scale=intrinsic_variance, space_mode=True)

print_metrics(noisy_sensor_clusters, metric_list_net_intrinsic, intrinsic_dim=dim_intrinsic, titleStr="Net Learned Intrinsic Jacobians", scale=intrinsic_variance, space_mode=False)

metric_list_def, metric_list_full = get_metrics_from_points(noisy_sensor_clusters, noisy_sensor_base, noisy_sensor_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance)

print_metrics(noisy_sensor_clusters, metric_list_def, intrinsic_dim=dim_intrinsic, titleStr="Locally Learned Tangent Jacobians", scale=intrinsic_variance, space_mode=True)

print_metrics(noisy_sensor_clusters, metric_list_def, intrinsic_dim=dim_intrinsic, titleStr="Locally Learned Intrinsic Jacobians", scale=intrinsic_variance, space_mode=False)


dist_mat_true_squared = calc_dist(intrinsic_process_clusters)
dist_mat_measured_squared = calc_dist(noisy_sensor_clusters)
dist_mat_local_squared = calc_dist(noisy_sensor_clusters, metric_list_def)
dist_mat_local_full_squared = calc_dist(noisy_sensor_clusters, metric_list_full)
dist_mat_net_tangent_squared = calc_dist(noisy_sensor_clusters, metric_list_net_tangent)
dist_mat_net_intrinsic_squared = calc_dist(noisy_sensor_clusters, metric_list_net_intrinsic)

dist_mat_true = numpy.sqrt(dist_mat_true_squared)
dist_mat_measured = numpy.sqrt(dist_mat_measured_squared)
dist_mat_local = numpy.sqrt(dist_mat_local_squared)
dist_mat_local_full = numpy.sqrt(dist_mat_local_full_squared)
dist_mat_net_tangent = numpy.sqrt(dist_mat_net_tangent_squared)
dist_mat_net_intrinsic = numpy.sqrt(dist_mat_net_intrinsic_squared)


#Keep only best distances
dist_mat_measured_trimmed = trim_distances(dist_mat_measured, n_neighbors=n_neighbors_mds)
dist_mat_measured_geo = scipy.sparse.csgraph.shortest_path(dist_mat_measured_trimmed, directed=False)


#dist_mat_net_tangent_trimmed = trim_distances(dist_mat_net_tangent, dist_mat_local_full, n_neighbors=n_neighbors_mds)
dist_mat_net_intrinsic_trimmed = trim_distances(dist_mat_net_intrinsic, dist_mat_true, n_neighbors=n_neighbors_mds)
dist_mat_local_trimmed = trim_distances(dist_mat_local, dist_mat_true, n_neighbors=n_neighbors_mds)
dist_mat_true_trimmed = trim_distances(dist_mat_true, n_neighbors=n_neighbors_mds)


#Geodesicly Complete Distances
#dist_mat_net_tangent_geo = scipy.sparse.csgraph.shortest_path(dist_mat_net_tangent, directed=False)
dist_mat_net_intrinsic_geo = scipy.sparse.csgraph.shortest_path(dist_mat_net_intrinsic_trimmed, directed=False)
dist_mat_local_geo = scipy.sparse.csgraph.shortest_path(dist_mat_local_trimmed, directed=False)
dist_mat_true_geo = scipy.sparse.csgraph.shortest_path(dist_mat_true_trimmed, directed=False)

#Reclustering

n_points_used_for_clusters_2 = min(n_points_used_for_clusters, n_points_used_for_clusters_2)
n_points_used_for_clusters_indexs_2 = numpy.random.choice(n_points_used_for_clusters, size=n_points_used_for_clusters_2, replace=False)
intrinsic_process_clusters_2 = intrinsic_process_clusters[:, n_points_used_for_clusters_indexs_2]
noisy_sensor_clusters_2 = noisy_sensor_clusters[:, n_points_used_for_clusters_indexs_2]
n_points_used_for_clusters_2 = intrinsic_process_clusters_2.shape[1]

color_map_clusters_2 = color_map_clusters[n_points_used_for_clusters_indexs_2, :]

dist_mat_local_geo = dist_mat_local_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
dist_mat_true_geo = dist_mat_true_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
dist_mat_measured_geo = dist_mat_measured_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
#dist_mat_net_tangent_geo = dist_mat_net_tangent_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
dist_mat_net_intrinsic_geo = dist_mat_net_intrinsic_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]

#dist_mat_local_trimmed = trim_distances(dist_mat_local_geo, dist_mat_local_full, n_neighbors=n_neighbors_mds)
dist_mat_local_trimmed = dist_mat_local_trimmed[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
#dist_mat_net_intrinsic_trimmed = trim_distances(dist_mat_net_intrinsic_geo, dist_mat_local_full, n_neighbors=n_neighbors_mds)
dist_mat_net_intrinsic_trimmed = dist_mat_net_intrinsic_trimmed[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]

dist_mat_true_trimmed = trim_distances(dist_mat_true, dist_mat_true, n_neighbors=n_neighbors_mds)


mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)


iso_embedding_local = intrinsic_isomaps(dist_mat_local_geo, dist_mat_local_trimmed, dim_intrinsic)
iso_embedding_net_intrinsic = intrinsic_isomaps(dist_mat_net_intrinsic_geo, dist_mat_net_intrinsic_trimmed, dim_intrinsic)
iso_embedding_true_sp = mds.fit(dist_mat_true_geo).embedding_
iso_embedding_measured_sp = mds.fit(dist_mat_measured_geo).embedding_
#iso_embedding_true = mds.fit(dist_mat_true_geo).embedding_
#iso_embedding_measured = mds.fit(dist_mat_measured_geo).embedding_


#iso_embedding_true_sp = mds.fit(dist_mat_true_flat, weight=dist_mat_true_flat_wgt).embedding_
#print(mds.stress_)
#iso_embedding_true_sp = mds.fit(dist_mat_true_trimmed, weight=(dist_mat_true_trimmed!=0).astype(int), init=iso_embedding_true_sp).embedding_
#print(mds.stress_)


#diff_embedding_tangent = calc_diff_map(dist_mat_net_tangent, dims=2, factor=2)
#diff_embedding_intrinsic = calc_diff_map(dist_mat_net_intrinsic, dims=2, factor=2)
#diff_embedding_local = calc_diff_map(dist_mat_local_geo, dims=2, factor=2)
#diff_embedding_true_sp = calc_diff_map(dist_mat_true_geo, dims=2, factor=2)
#diff_embedding_measured_sp = calc_diff_map(dist_mat_measured_geo, dims=2, factor=2)
#diff_embedding_true = calc_diff_map(iso_embedding_true, dims=2, factor=2)
#diff_embedding_measured = calc_diff_map(dist_mat_measured, dims=2, factor=2)


###Isomaps##############################################################################################################
print_process(iso_embedding_net_intrinsic.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Net-Learned Intrinsic Metric", align_points=intrinsic_process_clusters_2)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_net_intrinsic.T, titleStr="Isomap with Net-Learned Intrinsic Metric", n_points=n_points_used_for_clusters_2)
print('iso_embedding_intrinsic:', stress_normlized)

print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters_2)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_local.T, titleStr="Isomap with Locally Learned Intrinsic Metric", n_points=n_points_used_for_clusters_2)
print('iso_embedding_local:', stress_normlized)

print_process(iso_embedding_true_sp.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Exact Short Distances", align_points=intrinsic_process_clusters_2)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_true_sp.T, titleStr="Isomap with Exact Short Distances", n_points=n_points_used_for_clusters_2)
print('iso_embedding_true_sp:', stress_normlized)

print_process(iso_embedding_measured_sp.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Regular Isomap", align_points=intrinsic_process_clusters_2)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_measured_sp.T, titleStr="Regular Isomap", n_points=n_points_used_for_clusters_2)
print('iso_embedding_measured_sp:', stress_normlized)

#print_process(iso_embedding_true.T, bounding_shape=None, color_map=color_map_clusters, titleStr="PCA on Intrinsic Space")
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_true.T, titleStr="PCA on Intrinsic Space", n_points=n_points_used_for_clusters)
#print('iso_embedding_true:', stress_normlized)

#print_process(iso_embedding_measured.T, bounding_shape=None, color_map=color_map_clusters, titleStr="PCA on Measured Space")
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_measured.T, titleStr="PCA on Measured Space", n_points=n_points_used_for_clusters)
#print('iso_embedding_measured:', stress_normlized)
########################################################################################################################

###Diffusion Maps#######################################################################################################
#print_process(diff_embedding_intrinsic.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps with Net-Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_intrinsic.T, titleStr="Diffusion Maps with Net-Learned Intrinsic Metric", n_points=n_points_used_for_clusters)
#print('diff_embedding_intrinsic:', stress_normlized)

#print_process(diff_embedding_tangent.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps with Net-Learned Tangent Metric", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_tangent.T, titleStr="Diffusion Maps with Net-Learned Tangent Metric", n_points=n_points_used_for_clusters)
#print('diff_embedding_tangent:', stress_normlized)

#print_process(diff_embedding_local.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_local.T, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")
#print('diff_embedding_local:', stress_normlized)

#print_process(diff_embedding_true_sp.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps with Exact Short Distances", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_true_sp.T, titleStr="Diffusion Maps with Exact Short Distances")
#print('diff_embedding_true_sp:', stress_normlized)

#print_process(diff_embedding_true.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps on Intrinsic Space", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_true.T, titleStr="Diffusion Maps on Intrinsic Space", n_points=n_points_used_for_clusters)
#print('diff_embedding_true:', stress_normlized)

#print_process(diff_embedding_measured_sp.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Regular Diffusion Maps", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_measured_sp.T, titleStr="Regular Diffusion Maps")
#print('diff_embedding_measured_sp:', stress_normlized)

#print_process(diff_embedding_measured.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Diffusion Maps on Measured Space", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, diff_embedding_measured.T, titleStr="Diffusion Maps on Measured Space", n_points=n_points_used_for_clusters)
#print('diff_embedding_measured:', stress_normlized)
########################################################################################################################
print("Finish")
plt.show(block=True)
