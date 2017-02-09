from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import print_process, create_color_map, print_dynamics
from Util import *
import numpy

from non_local_tangent import non_local_tangent_net
###Settings#############################################################################################################
sim_dir_name = "2D Unit Square - Triangulation" #Which dataset to run
process_mode = "Static"

n_points_used_for_dynamics = 2000 #How many points are available from which to infer dynamics
n_points_used_for_plotting_dynamics = 400
n_points_used_for_clusters = 1000 #How many cluster to use in Kernal method
n_points_used_for_clusters_2 = 400 #How many cluster to use in Kernal method

n_neighbors_cov = 40 #How neighboors to use from which to infer dynamics locally
n_neighbors_mds = 20 #How many short distances are kept for each cluster point
n_hidden_drift = 4 #How many nodes in hidden layer that learns intrinsic dynamics
n_hidden_tangent = 20 #How many nodes in hidden layer that learns tangent plane
n_hidden_int = 20 #How many nodes in hidden layer that learns intrinsic dynamics
########################################################################################################################

sim_dir = './' + sim_dir_name
dtype = numpy.float64

intrinsic_process = numpy.loadtxt(sim_dir + '/' + 'intrinsic_used.txt', delimiter=',', dtype=dtype).T
noisy_sensor_measured = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy.txt', delimiter=',', dtype=dtype).T
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=dtype)
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy').astype(dtype=dtype)
dist_potential = numpy.loadtxt(sim_dir + '/' + 'dist_potential_used.txt', delimiter=',', dtype=dtype)

dim_intrinsic = intrinsic_process.shape[0]
dim_measurement = noisy_sensor_measured.shape[0]
n_points = intrinsic_process.shape[1]


if process_mode=="Static":
    noisy_sensor = noisy_sensor_measured[:, ::(dim_intrinsic+1)]
elif process_mode=="Dynamic":
    noisy_sensor = noisy_sensor_measured[:, ::2]
else:
    assert(0)

#measurement_variance = 0.000001


#ts = time.time()
#ts = round(ts)
#dir_name = '/{}'.format(ts)
#full_dir_name = './' + 'Runs' + '/' + sim_dir_name + dir_name
#os.makedirs(full_dir_name)

noisy_sensor_mean = noisy_sensor.mean()
noisy_sensor = (noisy_sensor - noisy_sensor_mean)

n_points_used_for_dynamics = min(n_points, n_points_used_for_dynamics)
points_used_dynamics_index = numpy.random.choice(n_points, size=n_points_used_for_dynamics, replace=False)

intrinsic_process = intrinsic_process[:, points_used_dynamics_index]
noisy_sensor = noisy_sensor[:, points_used_dynamics_index]
dist_potential = dist_potential[points_used_dynamics_index]

if process_mode == "Static":
    noisy_sensor_measured = noisy_sensor_measured.T
    noisy_sensor_measured_new = numpy.zeros((points_used_dynamics_index.shape[0]*(dim_intrinsic+1),dim_measurement))
    for i_index in range(points_used_dynamics_index.shape[0]):
        noisy_sensor_measured_new[i_index*(dim_intrinsic+1):(i_index+1)*(dim_intrinsic+1), :] = noisy_sensor_measured[points_used_dynamics_index[i_index]*(dim_intrinsic+1):(points_used_dynamics_index[i_index]+1)*(dim_intrinsic+1),:]
    noisy_sensor_measured = noisy_sensor_measured_new.T
elif process_mode == "Dynamics":
    noisy_sensor_measured = noisy_sensor_measured.T
    noisy_sensor_measured_new = numpy.zeros(noisy_sensor_measured.shape)
    for i_index in range(points_used_dynamics_index.shape[0]):
        noisy_sensor_measured_new[i_index * (2):(i_index + 1) * (2),:] = noisy_sensor_measured[points_used_dynamics_index[i_index] * (2):(points_used_dynamics_index[i_index] + 1) * (2), :]
    noisy_sensor_measured = noisy_sensor_measured_new.T
else:
    assert()


n_points = intrinsic_process.shape[1]

n_points_used_for_plotting_dynamics = min(n_points, n_points_used_for_plotting_dynamics)
points_dynamics_plot_index = numpy.random.choice(n_points, size=n_points_used_for_plotting_dynamics, replace=False)

color_map = create_color_map(intrinsic_process)

print_process(intrinsic_process, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Space")
#plt.savefig(full_dir_name + '/' + 'intrinsic_base.png', bbox_inches='tight')

print_process(noisy_sensor, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Space")

fig = plt.figure()
ax = fig.gca()
ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c=noisy_sensor[0, :])
plt.title("Sensor 1")

fig = plt.figure()
ax = fig.gca()
ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c=noisy_sensor[1, :])
plt.title("Sensor 2")

fig = plt.figure()
ax = fig.gca()
ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c=noisy_sensor[2, :])
plt.title("Sensor 3")
#plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')

#print_dynamics(intrinsic_process_base, intrinsic_process_step, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Process Dynamics")
#plt.savefig(full_dir_name + '/' + 'intrinsic_dynamics.png', bbox_inches='tight')

#print_dynamics(noisy_sensor_base, noisy_sensor_step, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Process Dynamics")
#plt.savefig(full_dir_name + '/' + 'sensor_dynamics.png', bbox_inches='tight')

###Intrinsic Metric Learning Net########################################################################################
#non_local_tangent_net_instance = non_local_tangent_net(intrinsic_process_base, dim_measurements=dim_measurement, dim_intrinsic=dim_intrinsic, n_hidden_tangent=n_hidden_tangent,  n_hidden_int=n_hidden_int, intrinsic_variance=intrinsic_variance, measurement_variance=measurement_variance)
#non_local_tangent_net_instance.train_net(noisy_sensor_base, noisy_sensor_step)
########################################################################################################################

#Testing and comparison with other methods##############################################################################
n_points_used_for_clusters = min(n_points, n_points_used_for_clusters)
points_used_for_clusters_indexs = numpy.random.choice(n_points, size=n_points_used_for_clusters, replace=False)

intrinsic_process_clusters = intrinsic_process[:, points_used_for_clusters_indexs]
noisy_sensor_clusters = noisy_sensor[:, points_used_for_clusters_indexs]
dist_potential_clusters = dist_potential[points_used_for_clusters_indexs]

if process_mode == "Static":
    noisy_sensor_measured = noisy_sensor_measured.T
    noisy_sensor_measured_new = numpy.zeros((points_used_for_clusters_indexs.shape[0]*(dim_intrinsic+1),dim_measurement))
    for i_index in range(points_used_for_clusters_indexs.shape[0]):
        noisy_sensor_measured_new[i_index*(dim_intrinsic+1):(i_index+1)*(dim_intrinsic+1), :] = noisy_sensor_measured[points_used_for_clusters_indexs[i_index]*(dim_intrinsic+1):(points_used_for_clusters_indexs[i_index]+1)*(dim_intrinsic+1),:]
    noisy_sensor_measured = noisy_sensor_measured_new.T
elif process_mode == "Dynamics":
    noisy_sensor_measured = noisy_sensor_measured.T
    noisy_sensor_measured_new = numpy.zeros(noisy_sensor_measured.shape)
    for i_index in range(points_used_for_clusters_indexs.shape[0]):
        noisy_sensor_measured_new[i_index * (2):(i_index + 1) * (2),:] = noisy_sensor_measured[points_used_for_clusters_indexs[i_index] * (2):(points_used_for_clusters_indexs[i_index] + 1) * (2), :]
    noisy_sensor_measured = noisy_sensor_measured_new.T
else:
    assert()


n_points_used_for_clusters = intrinsic_process_clusters.shape[1]


color_map_clusters = color_map[points_used_for_clusters_indexs, :]

test_ml(noisy_sensor_clusters, intrinsic_process_clusters, n_neighbors=n_neighbors_mds, n_components=dim_intrinsic, color=color_map_clusters)


#metric_list_net_tangent, metric_list_net_intrinsic = get_metrics_from_net(non_local_tangent_net_instance, noisy_sensor_clusters)

#net_drift = get_drift_from_net(non_local_tangent_net_instance, noisy_sensor_clusters)



#print_drift(noisy_sensor_clusters, net_drift, titleStr="Net Learned Drift")

#print_metrics(noisy_sensor_clusters, metric_list_net_tangent, intrinsic_dim=dim_intrinsic, titleStr="Net Learned Tangent Space", scale=intrinsic_variance*0.1, space_mode=False)

#print_metrics(noisy_sensor_clusters, metric_list_net_intrinsic, intrinsic_dim=dim_intrinsic, titleStr="Net Learned Intrinsic Jacobians", scale=intrinsic_variance, space_mode=False)

if process_mode=="Static":
    metric_list_def, metric_list_full = get_metrics_from_points_static(noisy_sensor_measured, dim_intrinsic, intrinsic_variance, measurement_variance)
else:
    noisy_sensor_base = noisy_sensor_measured [:, ::2]
    noisy_sensor_step = noisy_sensor_measured [:, 1::2]
    metric_list_def, metric_list_full = get_metrics_from_points(noisy_sensor_clusters, noisy_sensor_base, noisy_sensor_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance, measurement_variance)


print_metrics(noisy_sensor_clusters, metric_list_def, intrinsic_dim=dim_intrinsic, titleStr="Locally Learned Tangent Jacobians", scale=0.0005, space_mode=True, elipse=True, color_map=color_map_clusters)

print_metrics(noisy_sensor_clusters, metric_list_def, intrinsic_dim=dim_intrinsic, titleStr="Locally Learned Intrinsic Jacobians", scale=36*intrinsic_variance, space_mode=False, elipse=True, color_map=color_map_clusters)

#plt.show(block=True)


dist_mat_true_squared = calc_dist(intrinsic_process_clusters)
dist_mat_measured_squared = calc_dist(noisy_sensor_clusters)
dist_mat_local_squared = calc_dist(noisy_sensor_clusters, metric_list_def)
dist_mat_local_full_squared = calc_dist(noisy_sensor_clusters, metric_list_full)
#dist_mat_net_tangent_squared = calc_dist(noisy_sensor_clusters, metric_list_net_tangent)
#dist_mat_net_intrinsic_squared = calc_dist(noisy_sensor_clusters, metric_list_net_intrinsic)

dist_mat_true = numpy.sqrt(dist_mat_true_squared)
dist_mat_measured = numpy.sqrt(dist_mat_measured_squared)
dist_mat_local = numpy.sqrt(dist_mat_local_squared)
dist_mat_local_full = numpy.sqrt(dist_mat_local_full_squared)
#dist_mat_net_tangent = numpy.sqrt(dist_mat_net_tangent_squared)
#dist_mat_net_intrinsic = numpy.sqrt(dist_mat_net_intrinsic_squared)


#Keep only best distances
dist_mat_measured_trimmed = trim_distances(dist_mat_measured, n_neighbors=n_neighbors_mds)
dist_mat_measured_geo = scipy.sparse.csgraph.shortest_path(dist_mat_measured_trimmed, directed=False)


#dist_mat_net_tangent_trimmed = trim_distances(dist_mat_net_tangent, dist_mat_local_full, n_neighbors=n_neighbors_mds)
#dist_mat_net_intrinsic_trimmed = trim_distances(dist_mat_net_intrinsic, dist_mat_true, n_neighbors=n_neighbors_mds)
dist_mat_local_trimmed = trim_distances(dist_mat_local, dist_mat_local_full, n_neighbors=n_neighbors_mds)
dist_mat_true_trimmed = trim_distances(dist_mat_true, n_neighbors=n_neighbors_mds)


#Geodesicly Complete Distances
#dist_mat_net_tangent_geo = scipy.sparse.csgraph.shortest_path(dist_mat_net_tangent, directed=False)
#dist_mat_net_intrinsic_geo = scipy.sparse.csgraph.shortest_path(dist_mat_net_intrinsic_trimmed, directed=False)
dist_mat_local_geo = scipy.sparse.csgraph.shortest_path(dist_mat_local_trimmed, directed=False)
dist_mat_true_geo = scipy.sparse.csgraph.shortest_path(dist_mat_true_trimmed, directed=False)

#Reclustering

n_points_used_for_clusters_2 = min(n_points_used_for_clusters, n_points_used_for_clusters_2)
points_used_for_clusters_indexs_2 = numpy.random.choice(n_points_used_for_clusters, size=n_points_used_for_clusters_2, replace=False)
intrinsic_process_clusters_2 = intrinsic_process_clusters[:, points_used_for_clusters_indexs_2]
noisy_sensor_clusters_2 = noisy_sensor_clusters[:, points_used_for_clusters_indexs_2]
dist_potential_2 = dist_potential_clusters[points_used_for_clusters_indexs_2]
n_points_used_for_clusters_2 = intrinsic_process_clusters_2.shape[1]

color_map_clusters_2 = color_map_clusters[points_used_for_clusters_indexs_2, :]

dist_mat_local_geo = dist_mat_local_geo[points_used_for_clusters_indexs_2, :][:,points_used_for_clusters_indexs_2]
dist_mat_true_geo = dist_mat_true_geo[points_used_for_clusters_indexs_2, :][:,points_used_for_clusters_indexs_2]
dist_mat_measured_geo = dist_mat_measured_geo[points_used_for_clusters_indexs_2, :][:,points_used_for_clusters_indexs_2]
dist_mat_local_full = dist_mat_local_full[points_used_for_clusters_indexs_2, :][:,points_used_for_clusters_indexs_2]
dist_mat_true = dist_mat_true[points_used_for_clusters_indexs_2, :][:,points_used_for_clusters_indexs_2]

#dist_mat_net_tangent_geo = dist_mat_net_tangent_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
#dist_mat_net_intrinsic_geo = dist_mat_net_intrinsic_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]

dist_mat_local_trimmed = trim_distances(dist_mat_local_geo, n_neighbors=n_neighbors_mds)
dist_mat_local_trimmed_topo = trim_distances_topo(trim_distances(dist_mat_local_geo, n_neighbors=n_neighbors_mds) , dist_potential=dist_potential_2, radius_trim=0.4, intrinsic_process=intrinsic_process_clusters_2)

#dist_mat_local_trimmed = dist_mat_local_trimmed[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
#dist_mat_net_intrinsic_trimmed = trim_distances(dist_mat_net_intrinsic_geo, dist_mat_true, n_neighbors=n_neighbors_mds)
#dist_mat_net_intrinsic_trimmed = dist_mat_net_intrinsic_trimmed[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]

dist_mat_true_trimmed = trim_distances(dist_mat_true, n_neighbors=n_neighbors_mds)


mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)


#iso_embedding_local = intrinsic_isomaps(dist_mat_local_geo, dist_mat_local_trimmed, dim_intrinsic, intrinsic_process_clusters_2)
iso_embedding_local = mds.fit(dist_mat_local_geo).embedding_
iso_embedding_local_topo = mds.fit((1/2)*(dist_mat_local_trimmed_topo+dist_mat_local_trimmed_topo.T), weight=(dist_mat_local_trimmed_topo!=0).astype(int)).embedding_

#iso_embedding_net_intrinsic = intrinsic_isomaps(dist_mat_net_intrinsic_geo, dist_mat_net_intrinsic_trimmed, dim_intrinsic, intrinsic_process_clusters_2)
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
#print_process(iso_embedding_net_intrinsic.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Net-Learned Intrinsic Metric", align_points=intrinsic_process_clusters_2)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_net_intrinsic.T, titleStr="Isomap with Net-Learned Intrinsic Metric", n_points=n_points_used_for_clusters_2)
#print('iso_embedding_intrinsic:', stress_normlized)

print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters_2)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_local.T, titleStr="Isomap with Locally Learned Intrinsic Metric", n_points=n_points_used_for_clusters_2)
print('iso_embedding_local:', stress_normlized)

print_process(iso_embedding_local_topo.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Locally Learned Intrinsic Metric and Topo", align_points=intrinsic_process_clusters_2)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_local_topo.T, titleStr="Isomap with Locally Learned Intrinsic Metric and Topo", n_points=n_points_used_for_clusters_2)
print('iso_embedding_local_topo:', stress_normlized)

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
