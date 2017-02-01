from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import print_process, create_color_map, print_dynamics
from Util import *
import numpy

###Settings#############################################################################################################
sim_dir_name = "2D Room - Exact Limits" #Which dataset to run

n_points_used_for_dynamics = 50000 #How many points are available from which to infer dynamics
n_points_used_for_plotting_dynamics = 500
n_points_used_for_clusters = 8000 #How many cluster to use in Kernal method
n_points_used_for_clusters_2 = 3000 #How many cluster to use in Kernal method

n_neighbors_cov = 20 #How neighboors to use from which to infer dynamics locally
n_neighbors_mds = 20 #How many short distances are kept for each cluster point
n_hidden_drift = 4 #How many nodes in hidden layer that learns intrinsic dynamics
n_hidden_tangent = 20 #How many nodes in hidden layer that learns tangent plane
n_hidden_int = 20 #How many nodes in hidden layer that learns intrinsic dynamics
########################################################################################################################

sim_dir = './' + sim_dir_name
dtype = numpy.float64

intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',', dtype=dtype).T
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',', dtype=dtype).T
noisy_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_base.txt', delimiter=',', dtype=dtype)/100
noisy_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_step.txt', delimiter=',', dtype=dtype)/100
noisy_sensor_base_mean = noisy_sensor_base.mean()
noisy_sensor_base = (noisy_sensor_base - noisy_sensor_base_mean)
noisy_sensor_step = (noisy_sensor_step - noisy_sensor_base_mean)
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=dtype)
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy').astype(dtype=dtype)

#measurement_variance = 0.001
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
#non_local_tangent_net_instance = non_local_tangent_net(intrinsic_process_base, dim_measurements=dim_measurement, dim_intrinsic=dim_intrinsic, n_hidden_drift=n_hidden_drift, n_hidden_tangent=n_hidden_tangent,  n_hidden_int=n_hidden_int, intrinsic_variance=intrinsic_variance, measurement_variance=measurement_variance)
#non_local_tangent_net_instance.train_net(noisy_sensor_base, noisy_sensor_step)
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

#metric_list_net_tangent, metric_list_net_intrinsic = get_metrics_from_net(non_local_tangent_net_instance, noisy_sensor_clusters)

#net_drift = get_drift_from_net(non_local_tangent_net_instance, noisy_sensor_clusters)



#print_drift(noisy_sensor_clusters, net_drift, titleStr="Net Learned Drift")

#print_metrics(noisy_sensor_clusters, metric_list_net_tangent, intrinsic_dim=dim_intrinsic, titleStr="Net Learned Tangent Space", scale=intrinsic_variance, space_mode=True)

#print_metrics(noisy_sensor_clusters, metric_list_net_intrinsic, intrinsic_dim=dim_intrinsic, titleStr="Net Learned Intrinsic Jacobians", scale=intrinsic_variance, space_mode=False)

metric_list_def, metric_list_full = get_metrics_from_points(noisy_sensor_clusters, noisy_sensor_base, noisy_sensor_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance)

#print_metrics(noisy_sensor_clusters, metric_list_def, intrinsic_dim=dim_intrinsic, titleStr="Locally Learned Tangent Jacobians", scale=intrinsic_variance, space_mode=True)

#print_metrics(noisy_sensor_clusters, metric_list_def, intrinsic_dim=dim_intrinsic, titleStr="Locally Learned Intrinsic Jacobians", scale=intrinsic_variance, space_mode=False)


#dist_mat_net_tangent = numpy.sqrt(calc_dist(noisy_sensor_clusters, metric_list_net_tangent))
#dist_mat_net_intrinsic = numpy.sqrt(calc_dist(noisy_sensor_clusters, metric_list_net_intrinsic))

#dist_mat_measured_KL = numpy.zeros((n_points_used_for_clusters,n_points_used_for_clusters))

#for i_point in range(n_points_used_for_clusters):
#    print(i_point)
#    for j_point in range(n_points_used_for_clusters):
#        dist_mat_measured_KL[i_point, j_point] = (1/2)*numpy.trace(numpy.dot(metric_list_full[i_point], numpy.linalg.inv(metric_list_full[j_point])))
#
#dist_mat_measured_KL = dist_mat_measured_KL + dist_mat_measured_KL.T

dist_mat_true_squared = calc_dist(intrinsic_process_clusters)
dist_mat_measured_squared = calc_dist(noisy_sensor_clusters)
dist_mat_local_squared = calc_dist(noisy_sensor_clusters, metric_list_def)
dist_mat_local_full_squared = calc_dist(noisy_sensor_clusters, metric_list_full)

dist_mat_true = numpy.sqrt(dist_mat_true_squared)
dist_mat_measured = numpy.sqrt(dist_mat_measured_squared)
dist_mat_local = numpy.sqrt(dist_mat_local_squared)
dist_mat_local_full = numpy.sqrt(dist_mat_local_full_squared)

#dist_mat_measured_KL = dist_mat_measured_KL+dist_mat_local_squared
#Keep only best distances

#dist_mat_net_tangent = trim_distances(dist_mat_net_tangent, dist_mat_measured, n_neighbors=n_neighbors_mds)
#dist_mat_net_intrinsic_trimmed = trim_distances(dist_mat_net_intrinsic, dist_mat_measured, n_neighbors=n_neighbors_mds)
dist_mat_measured_trimmed = trim_distances(dist_mat_measured, n_neighbors=10)
dist_mat_measured_geo = scipy.sparse.csgraph.shortest_path(dist_mat_measured_trimmed, directed=False)
dist_mat_local_trimmed = trim_distances(dist_mat_local, dist_mat_local_full, n_neighbors=n_neighbors_mds)
#dist_mat_true_trimmed = trim_distances(dist_mat_true, dist_mat_true, n_neighbors=n_neighbors_mds)


#Geodesicly Complete Distances

#dist_mat_net_tangent_geo = scipy.sparse.csgraph.shortest_path(dist_mat_net_tangent, directed=False)
#dist_mat_net_intrinsic_geo = scipy.sparse.csgraph.shortest_path(dist_mat_net_intrinsic_trimmed, directed=False)
dist_mat_local_geo = scipy.sparse.csgraph.shortest_path(dist_mat_local_trimmed, directed=False)
#dist_mat_true_geo = scipy.sparse.csgraph.shortest_path(dist_mat_true_trimmed, directed=False)


n_points_used_for_clusters_2 = min(n_points_used_for_clusters, n_points_used_for_clusters_2)
n_points_used_for_clusters_indexs_2 = numpy.random.choice(n_points_used_for_clusters, size=n_points_used_for_clusters_2, replace=False)


#Measured Points
noisy_sensor_clusters_2 = noisy_sensor_clusters[:, n_points_used_for_clusters_indexs_2]
#Ground Truth
intrinsic_process_clusters_2 = intrinsic_process_clusters[:, n_points_used_for_clusters_indexs_2]

n_points_used_for_clusters_2 = intrinsic_process_clusters_2.shape[1]


color_map_clusters_2 = color_map_clusters[n_points_used_for_clusters_indexs_2, :]

dist_mat_local_geo = dist_mat_local_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
#dist_mat_true_geo = dist_mat_true_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
dist_mat_measured_geo = dist_mat_measured_geo[n_points_used_for_clusters_indexs_2, :][:,n_points_used_for_clusters_indexs_2]
'''
stress = numpy.sqrt(numpy.sum(numpy.square(dist_mat_true_geo - dist_mat_local_geo)))
stress_norm = numpy.sqrt(numpy.sum(numpy.square(dist_mat_true_geo)))
stress_normalized = stress / stress_norm
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(dist_mat_true_geo.reshape((1, n_points_used_for_clusters_2 * n_points_used_for_clusters_2)),
           dist_mat_local_geo[:].reshape((1, n_points_used_for_clusters_2 * n_points_used_for_clusters_2)), s=1)
ax.plot([0, numpy.max(dist_mat_true_geo)], [0, numpy.max(dist_mat_true_geo)], c='r')
ax.set_xlim([0, numpy.max(dist_mat_true_geo) * 1.15])
ax.set_ylim([0, numpy.max(dist_mat_true_geo) * 1.15])
ax.set_xlabel('Ground Truth Distances')
ax.set_ylabel('Distance in Recovered Embedding')
ax.set_title('Mahalanobis vs Real Dist')
plt.axis('equal')
plt.show(block=False)
'''
#dist_mat_local_geo, dist_mat_local_flat_wgt = trim_non_euc(dist_mat_local_trimmed, dist_mat_local_geo,  dim_intrinsic, intrinsic_process_clusters)

'''
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.show(block=False)
ax.scatter(noisy_sensor_clusters[0, :], noisy_sensor_clusters[1, :], noisy_sensor_clusters[2, :], c='k')
for i_point in range(n_points_used_for_clusters):
    for j_point in range(n_points_used_for_clusters):
        if dist_mat_true_sp[i_point, j_point] != 0:
            u = noisy_sensor_clusters[:, j_point] - noisy_sensor_clusters[:, i_point]
            ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point], noisy_sensor_clusters[2, i_point], u[0], u[1], u[2], length=numpy.linalg.norm(u), pivot='tail')
'''

'''
fig = plt.figure()
ax = fig.gca()
ax.scatter(intrinsic_process_clusters[0, :], intrinsic_process_clusters[1, :], c="k")
plt.show(block=False)
for i_point in range(n_points_used_for_clusters):
    for j_point in range(n_points_used_for_clusters):
        if dist_mat_local_wgt[i_point, j_point] != 0 :
            u = intrinsic_process_clusters[:, j_point] - intrinsic_process_clusters[:, i_point]
            ax.quiver(intrinsic_process_clusters[0, i_point], intrinsic_process_clusters[1, i_point], u[0], u[1], angles='xy', scale_units='xy', scale=1, pivot='tail', width=0.0005)

'''
mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)

#iso_embedding_intrinsic = mds.fit(dist_mat_net_intrinsic_geo).embedding_
#print(mds.stress_)
#iso_embedding_intrinsic = mds.fit(dist_mat_net_intrinsic_trimmed, weight=(dist_mat_net_intrinsic_trimmed!=0).astype(int), init=iso_embedding_intrinsic).embedding_
#print(mds.stress_)


D_squared = dist_mat_local_geo ** 2

# centering matrix
n = D_squared.shape[0]
J_c = 1. / n * (numpy.eye(n) - 1 + (n - 1) * numpy.eye(n))

# perform double centering
B = -0.5 * (J_c.dot(D_squared)).dot(J_c)

# find eigenvalues and eigenvectors
U, eigen_val, V = numpy.linalg.svd(B)
eigen_vect = V
eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])
eigen_vect = eigen_vect[eigen_val_sort_ind]
eigen_vect = eigen_vect[:dim_intrinsic].T

mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)
iso_embedding_local = mds.fit(dist_mat_local_geo, init=eigen_vect).embedding_
print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_local.T, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")
mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)
dist_mat_local_trimmed = trim_distances(dist_mat_local_geo, dist_mat_local_geo, n_neighbors=n_neighbors_mds)

iso_embedding_local = mds.fit(dist_mat_local_trimmed, weight=(dist_mat_local_trimmed!=0).astype(int), init=iso_embedding_local).embedding_
print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_local.T, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")

'''
D_squared = dist_mat_true_geo ** 2

# centering matrix
n = D_squared.shape[0]
J_c = 1. / n * (numpy.eye(n) - 1 + (n - 1) * numpy.eye(n))

# perform double centering
B = -0.5 * (J_c.dot(D_squared)).dot(J_c)

# find eigenvalues and eigenvectors
U, eigen_val, V = numpy.linalg.svd(B)
eigen_vect = V
eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])
eigen_vect = eigen_vect[eigen_val_sort_ind]
eigen_vect = eigen_vect[:dim_intrinsic].T

mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)
iso_embedding_true = mds.fit(dist_mat_true_geo, init=eigen_vect).embedding_
print_process(iso_embedding_true.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_true.T, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")
mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)
iso_embedding_true = mds.fit(dist_mat_true_trimmed, weight=(dist_mat_true_trimmed!=0).astype(int), init=iso_embedding_true).embedding_
print_process(iso_embedding_true.T, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_true.T, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")
'''

#print(mds.stress_)
#iso_embedding_local = mds.fit(dist_mat_local_trimmed, weight=(dist_mat_local_trimmed!=0).astype(int)).embedding_

#iso_embedding_true_sp = mds.fit(dist_mat_true_sp_temp, weight=dist_mat_true_sp_wgt).embedding_
#iso_embedding_true_sp = mds.fit(dist_mat_true_flat, weight=dist_mat_true_flat_wgt).embedding_
#print(mds.stress_)
#iso_embedding_true_sp = mds.fit(dist_mat_true_trimmed, weight=(dist_mat_true_trimmed!=0).astype(int), init=iso_embedding_true_sp).embedding_
#print(mds.stress_)

#iso_embedding_measured_sp = mds.fit(dist_mat_measured_geo).embedding_
#iso_embedding_true = mds.fit(dist_mat_true).embedding_
#iso_embedding_measured = mds.fit(dist_mat_measured).embedding_


#diff_embedding_tangent = calc_diff_map(dist_mat_net_tangent, dims=2, factor=2)
#diff_embedding_intrinsic = calc_diff_map(dist_mat_net_intrinsic, dims=2, factor=2)
#diff_embedding_local = calc_diff_map(dist_mat_local_geo, dims=2, factor=2)
#diff_embedding_true_sp = calc_diff_map(dist_mat_true_geo, dims=2, factor=2)
#diff_embedding_measured_sp = calc_diff_map(dist_mat_measured_geo, dims=2, factor=2)
#diff_embedding_true = calc_diff_map(iso_embedding_true, dims=2, factor=2)
#diff_embedding_measured = calc_diff_map(dist_mat_measured, dims=2, factor=2)


###Isomaps##############################################################################################################
#print_process(iso_embedding_intrinsic.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Isomap with Net-Learned Intrinsic Metric")
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_intrinsic.T, titleStr="Isomap with Net-Learned Intrinsic Metric", n_points=n_points_used_for_clusters)
#print('iso_embedding_intrinsic:', stress_normlized)

#print_process(iso_embedding_tangent.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Isomap with Net-Learned Tangent Metric")
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_tangent.T, titleStr="Isomap with Net-Learned Tangent Metric", n_points=n_points_used_for_clusters)
#print('iso_embedding_tangent:', stress_normlized)

#print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_local.T, titleStr="Isomap with Locally Learned Intrinsic Metric")
#print('iso_embedding_local:', stress_normlized)

#print_process(iso_embedding_true_sp.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Isomap with Exact Short Distances", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_true_sp.T, titleStr="Isomap with Exact Short Distances")
#print('iso_embedding_true_sp:', stress_normlized)

#print_process(iso_embedding_measured_sp.T, bounding_shape=None, color_map=color_map_clusters, titleStr="Regular Isomap", align_points=intrinsic_process_clusters)
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_measured_sp.T, titleStr="Regular Isomap")
#print('iso_embedding_measured_sp:', stress_normlized)


#print_process(iso_embedding_true.T, bounding_shape=None, color_map=color_map_clusters, titleStr="PCA on Intrinsic Space")
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_true.T, titleStr="PCA on Intrinsic Space", n_points=n_points_used_for_clusters)
#print('iso_embedding_true:', stress_normlized)

#print_process(iso_embedding_measured.T, bounding_shape=None, color_map=color_map_clusters, titleStr="PCA on Measured Space")
#stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_measured.T, titleStr="PCA on Measured Space", n_points=n_points_used_for_clusters)
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
