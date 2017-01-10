from __future__ import print_function
from __future__ import absolute_import
import time
import os
import downhill
import numpy
import matplotlib.pyplot as plt
from non_local_tangent import non_local_tangent_net
from DataGeneration import print_process, create_color_map
import theano.tensor as T
import theano
from DataGeneration import print_process, print_drift, create_color_map
from Util import *
from sklearn import manifold
import numpy
from downhill.util import as_float

sim_dir_name = "Non Tangent"
sim_dir = './' + sim_dir_name

ts = time.time()
ts = round(ts)
dir_name = '/{}'.format(ts)
full_dir_name = './' + 'Runs' + '/' + sim_dir_name + dir_name
os.makedirs(full_dir_name)

precision = 'float64'


if precision == 'float32':
    dtype = numpy.float32
    input_base_Theano = T.fmatrix('input_base_Theano')
    input_step_Theano = T.fmatrix('input_step_Theano')
    input_coeff_Theano = T.fmatrix('input_step_Theano')
    learning_rate_Theano = T.scalar('learning_rate_Theano')
elif precision == 'float64':
    dtype = numpy.float64
    input_base_Theano = T.dmatrix('input_base_Theano')
    input_step_Theano = T.dmatrix('input_step_Theano')
    input_coeff_Theano = T.dmatrix('input_step_Theano')
    learning_rate_Theano = T.scalar('learning_rate_Theano')



T.config.floatX = precision


n_points_used = 1000
intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',', dtype=dtype)
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',', dtype=dtype)
noisy_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_base.txt', delimiter=',', dtype=dtype)
noisy_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_step.txt', delimiter=',', dtype=dtype)
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=dtype)
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy').astype(dtype=dtype)
#TODO

measurement_variance = 0.00000000001

n_points = intrinsic_process_base.shape[1]
dim_intrinsic = intrinsic_process_base.shape[0]
dim_measurement = noisy_sensor_base.shape[0]


n_points_used = min(n_points, n_points_used)
points_used_index = numpy.random.choice(n_points, size=n_points_used, replace=False)

intrinsic_process_base = intrinsic_process_base[:, points_used_index]
intrinsic_process_step = intrinsic_process_step[:, points_used_index]
noisy_sensor_base = noisy_sensor_base[:, points_used_index]
noisy_sensor_step = noisy_sensor_step[:, points_used_index]

n_points = intrinsic_process_base.shape[1]


n_plot_points = 2000
n_plot_points = min(n_points, n_plot_points)
points_plot_index = numpy.random.choice(n_points, size=n_plot_points, replace=False)

color_map = create_color_map(intrinsic_process_base)

print_process(noisy_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Measurement Process")
plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')

non_local_tangent_net_instance = non_local_tangent_net(intrinsic_process_base, dim_measurements=3, dim_intrinsic=2, n_hidden_tangent=10, n_hidden_int=10, intrinsic_variance=intrinsic_variance, measurement_variance=measurement_variance)

n_iterations = 1000

get_jacobian = theano.function(
    inputs=[input_base_Theano],
    outputs=[non_local_tangent_net_instance.get_jacobian(input_base_Theano)],
)

get_jacobian_int = theano.function(
    inputs=[input_base_Theano],
    outputs=[non_local_tangent_net_instance.get_jacobian_int(input_base_Theano)],
)


get_cost = theano.function(
    inputs=[input_base_Theano, input_step_Theano, input_coeff_Theano],
    outputs=[non_local_tangent_net_instance.get_cost(input_base_Theano, input_step_Theano, input_coeff_Theano)],
)

get_cost_rec = theano.function(
    inputs=[input_base_Theano, input_step_Theano, input_coeff_Theano],
    outputs=[non_local_tangent_net_instance.get_cost_rec(input_base_Theano, input_step_Theano, input_coeff_Theano)],
)



coeffs = numpy.random.randn(dim_intrinsic, n_points)


params = [non_local_tangent_net_instance.W_tangent_1, non_local_tangent_net_instance.W_tangent_2, non_local_tangent_net_instance.W_tangent_3, non_local_tangent_net_instance.b_tangent_1, non_local_tangent_net_instance.b_tangent_2, non_local_tangent_net_instance.b_tangent_3, non_local_tangent_net_instance.W_int_1, non_local_tangent_net_instance.W_int_2, non_local_tangent_net_instance.W_int_3, non_local_tangent_net_instance.b_int_1, non_local_tangent_net_instance.b_int_2, non_local_tangent_net_instance.b_int_3]
params2 = [non_local_tangent_net_instance.W_int_1, non_local_tangent_net_instance.W_int_2, non_local_tangent_net_instance.W_int_3, non_local_tangent_net_instance.b_int_1, non_local_tangent_net_instance.b_int_2, non_local_tangent_net_instance.b_int_3]


gparams = [T.grad(non_local_tangent_net_instance.get_cost(input_base_Theano, input_step_Theano, input_coeff_Theano), param) for param in params]
gparams2 = [T.grad(non_local_tangent_net_instance.get_cost(input_base_Theano, input_step_Theano, input_coeff_Theano), param) for param in params2]

updates = [
    (param, param - learning_rate_Theano * gparam)
    for param, gparam in zip(params, gparams)
    ]

updates2 = [
    (param, param - learning_rate_Theano * gparam)
    for param, gparam in zip(params2, gparams2)
    ]



# compile the MSGD step into a theano function
MSGD = theano.function(inputs=[input_base_Theano, input_step_Theano, input_coeff_Theano, learning_rate_Theano], outputs=non_local_tangent_net_instance.get_cost(input_base_Theano, input_step_Theano, input_coeff_Theano), updates=updates)
MSGD2 = theano.function(inputs=[input_base_Theano, input_step_Theano, input_coeff_Theano, learning_rate_Theano], outputs=non_local_tangent_net_instance.get_cost(input_base_Theano, input_step_Theano, input_coeff_Theano), updates=updates2)

cost_mean_last = float('Inf')


cost_term = []

'''
train_build = downhill.build(
    loss=non_local_tangent_net_instance.get_cost(input_base_Theano, input_step_Theano, input_coeff_Theano).mean(),
    params=[non_local_tangent_net_instance.W_tangent_1, non_local_tangent_net_instance.W_tangent_2, non_local_tangent_net_instance.W_tangent_3, non_local_tangent_net_instance.b_tangent_1, non_local_tangent_net_instance.b_tangent_2, non_local_tangent_net_instance.b_tangent_3, non_local_tangent_net_instance.W_int_1, non_local_tangent_net_instance.W_int_2, non_local_tangent_net_instance.W_int_3, non_local_tangent_net_instance.b_int_1, non_local_tangent_net_instance.b_int_2, non_local_tangent_net_instance.b_int_3],
    inputs=[input_base_Theano, input_step_Theano, input_coeff_Theano],
    algo='adam',
    monitor_gradients=False)

train = downhill.Dataset([noisy_sensor_base[:, :].T, noisy_sensor_step[:, :].T, coeffs[:, :].T], batch_size=1)
valid = downhill.Dataset([noisy_sensor_base[:, :].T, noisy_sensor_step[:, :].T, coeffs[:, :].T], batch_size=1)

for idx, [tm, vm] in enumerate(train_build.iterate(train=train, valid=valid, rms_halflife=n_points_used/120, momentum=0.8, learning_rate=1e-4, validate_every=100, patience=5, min_improvement=1e-2, max_updates=1000)):
    cost_1 = numpy.zeros((n_points), dtype=dtype)
    if not(idx % 10):
        for i_point in range(0, n_points):
            jacobian = get_jacobian(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)))[0]
            jacobian_int = get_jacobian_int(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)))[0]
            coeffs[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian)/measurement_variance+numpy.dot(jacobian_int.T, jacobian_int)/intrinsic_variance), jacobian.T), noisy_sensor_step[:, i_point]-noisy_sensor_base[:, i_point])/measurement_variance
            cost_1[i_point] = get_cost(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)).T,noisy_sensor_step[:, i_point].reshape((dim_measurement, 1)).T, coeffs[:, i_point].reshape((dim_intrinsic, 1)).T)[0]

    cost_mean_new = cost_1.mean()
    cost_term.append(cost_mean_new)

    if not (idx % 1):
        print('Iteration ', idx + 1, ':')
        print('pretrain inv train:', tm['loss'])
        print('pretrain inv valid:', vm['loss'])

        if vm['loss'] < 1e-9:
            break
'''


learning_rate = 100000

cost_term = []

for i_iteration in range(0, 500):
    cost_1 = numpy.zeros((n_points), dtype=dtype)
    cost_2 = numpy.zeros((n_points), dtype=dtype)
    #cost_rec = numpy.zeros((n_points), dtype=dtype)
    for i_point in range(0, n_points):
        jacobian = get_jacobian(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)))[0]
        jacobian_int = get_jacobian_int(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)))[0]
        #cost_1[i_point] = get_cost(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)),noisy_sensor_step[:, i_point].reshape((dim_measurement, 1)), coeffs[:, i_point].reshape((dim_intrinsic, 1)))[0]
        #cost_rec[i_point] = get_cost_rec(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)),noisy_sensor_step[:, i_point].reshape((dim_measurement, 1)), coeffs[:, i_point].reshape((dim_intrinsic, 1)))[0]
        coeffs[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian)/measurement_variance+numpy.dot(jacobian_int.T, jacobian_int)/intrinsic_variance), jacobian.T), noisy_sensor_step[:, i_point]-noisy_sensor_base[:, i_point])/measurement_variance
        cost_2[i_point] = MSGD(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)).T,noisy_sensor_step[:, i_point].reshape((dim_measurement, 1)).T, coeffs[:, i_point].reshape((dim_intrinsic, 1)).T, learning_rate)
    cost_mean_new = cost_2.mean()
    cost_term.append(cost_mean_new)

    #print(cost_1.mean())
    #print(cost_2.mean())
    print(cost_2.mean())
    print(learning_rate)
    if (cost_mean_new<cost_mean_last):
        learning_rate = learning_rate*1.1
    else:
        learning_rate = learning_rate*0.5
    cost_mean_last = cost_mean_new

    if learning_rate < 1e-20:
        break


fig = plt.figure()
ax = fig.gca(projection='3d')

for i_point in range(0, 50):
    jacobian = get_jacobian(noisy_sensor_base[:, points_plot_index[i_point]].reshape((dim_measurement, 1)))[0]
    jacobian_int = get_jacobian_int(noisy_sensor_base[:, points_plot_index[i_point]].reshape((dim_measurement, 1)))[0]
    jacobian_total = numpy.dot(jacobian, numpy.linalg.pinv(jacobian_int))
    #jacobian_total = jacobian
    [u, s, v] = numpy.linalg.svd(numpy.dot(jacobian_total, jacobian_total.T))
    u = 3*numpy.dot(u, numpy.diag(numpy.sqrt(s*intrinsic_variance)))
    ax.quiver(noisy_sensor_base[0, points_plot_index[i_point]], noisy_sensor_base[1, points_plot_index[i_point]], noisy_sensor_base[2, points_plot_index[i_point]], u[0, 0], u[1, 0], u[2, 0] ,length=numpy.linalg.norm(u[:, 0]), pivot='tail')
    ax.quiver(noisy_sensor_base[0, points_plot_index[i_point]], noisy_sensor_base[1, points_plot_index[i_point]], noisy_sensor_base[2, points_plot_index[i_point]], u[0, 1], u[1, 1], u[2, 1] ,length=numpy.linalg.norm(u[:, 1]),  pivot='tail')


learning_rate = 1000

for i_iteration in range(0, 500):
    cost_1 = numpy.zeros((n_points), dtype=dtype)
    cost_2 = numpy.zeros((n_points), dtype=dtype)
    #cost_rec = numpy.zeros((n_points), dtype=dtype)
    for i_point in range(0, n_points):
        jacobian = get_jacobian(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)))[0]
        jacobian_int = get_jacobian_int(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)))[0]
        #cost_1[i_point] = get_cost(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)),noisy_sensor_step[:, i_point].reshape((dim_measurement, 1)), coeffs[:, i_point].reshape((dim_intrinsic, 1)))[0]
        #cost_rec[i_point] = get_cost_rec(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)),noisy_sensor_step[:, i_point].reshape((dim_measurement, 1)), coeffs[:, i_point].reshape((dim_intrinsic, 1)))[0]
        coeffs[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian)/measurement_variance+numpy.dot(jacobian_int.T, jacobian_int)/intrinsic_variance), jacobian.T), noisy_sensor_step[:, i_point]-noisy_sensor_base[:, i_point])/measurement_variance
        cost_2[i_point] = MSGD2(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)).T,noisy_sensor_step[:, i_point].reshape((dim_measurement, 1)).T, coeffs[:, i_point].reshape((dim_intrinsic, 1)).T, learning_rate)

    cost_mean_new = cost_2.mean()
    cost_term.append(cost_mean_new)

    #print(cost_1.mean())
    #print(cost_2.mean())
    print(cost_mean_new)
    print(learning_rate)
    if (cost_mean_new<cost_mean_last):
        learning_rate = learning_rate*1.1
    else:
        learning_rate = learning_rate*0.5
    cost_mean_last = cost_mean_new
    if learning_rate < 1e-20:
        break


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cost_term[1:], c='b')
ax.set_xlabel('Epochs')
ax.set_ylabel('Cost')
ax.set_title("Cost vs Epochs")

fig = plt.figure()
ax = fig.gca(projection='3d')


for i_point in range(0, 50):
    jacobian = get_jacobian(noisy_sensor_base[:, points_plot_index[i_point]].reshape((dim_measurement, 1)))[0]
    jacobian_int = get_jacobian_int(noisy_sensor_base[:, points_plot_index[i_point]].reshape((dim_measurement, 1)))[0]
    jacobian_total = numpy.dot(jacobian, numpy.linalg.pinv(jacobian_int))
    #jacobian_total = jacobian
    [u, s, v] = numpy.linalg.svd(numpy.dot(jacobian_total, jacobian_total.T))
    u = 3*numpy.dot(u, numpy.diag(numpy.sqrt(s*intrinsic_variance)))
    ax.quiver(noisy_sensor_base[0, points_plot_index[i_point]], noisy_sensor_base[1, points_plot_index[i_point]], noisy_sensor_base[2, points_plot_index[i_point]], u[0, 0], u[1, 0], u[2, 0] ,length=numpy.linalg.norm(u[:, 0]), pivot='tail')
    ax.quiver(noisy_sensor_base[0, points_plot_index[i_point]], noisy_sensor_base[1, points_plot_index[i_point]], noisy_sensor_base[2, points_plot_index[i_point]], u[0, 1], u[1, 1], u[2, 1] ,length=numpy.linalg.norm(u[:, 1]),  pivot='tail')




##################################################################################################
n_points_used = 300
intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',', dtype=dtype)
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',', dtype=dtype)
noisy_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_base.txt', delimiter=',', dtype=dtype)
noisy_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_step.txt', delimiter=',', dtype=dtype)
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=dtype)
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy').astype(dtype=dtype)
#TODO

measurement_variance = 0.000000001

n_points = intrinsic_process_base.shape[1]
dim_intrinsic = intrinsic_process_base.shape[0]
dim_measurement = noisy_sensor_base.shape[0]


n_points_used = min(n_points, n_points_used)
points_used_index = numpy.random.choice(n_points, size=n_points_used, replace=False)

intrinsic_process_base = intrinsic_process_base[:, points_used_index]
intrinsic_process_step = intrinsic_process_step[:, points_used_index]
noisy_sensor_base = noisy_sensor_base[:, points_used_index]
noisy_sensor_step = noisy_sensor_step[:, points_used_index]

n_points = intrinsic_process_base.shape[1]


n_plot_points = 2000
n_plot_points = min(n_points, n_plot_points)
points_plot_index = numpy.random.choice(n_points, size=n_plot_points, replace=False)

color_map = create_color_map(intrinsic_process_base)

######################################################################################################
cov_list_def = [None] * n_points
cov_list_full = [None] * n_points

for i_point in range(0, n_points):
    jacobian = get_jacobian(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)))[0]
    jacobian_int = get_jacobian_int(noisy_sensor_base[:, i_point].reshape((dim_measurement, 1)))[0]
    jacobian_total = numpy.dot(jacobian, numpy.linalg.pinv(jacobian_int))
    temp_cov =  numpy.dot(jacobian_total, jacobian_total.T)
    U, s, V = numpy.linalg.svd(temp_cov)
    s_full = numpy.copy(s)
    s_def = numpy.copy(s)
    s_def[dim_intrinsic:] = float('Inf')
    s_def = 1 / s_def
    if s_def[dim_intrinsic:] < numpy.finfo(numpy.float32).eps:
        s_full[dim_intrinsic:] = numpy.finfo(numpy.float32).eps

    s_full = 1 / s_full
    cov_list_def[i_point] = numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
    cov_list_full[i_point] = numpy.dot(U, numpy.dot(numpy.diag(s_full), V))

approx_dist = numpy.zeros((n_points, n_points))
dist_mat_full = numpy.zeros((n_points, n_points))
dist_mat_def = numpy.zeros((n_points, n_points))
dist_mat_true = numpy.zeros((n_points, n_points))
dist_mat_measured = numpy.zeros((n_points, n_points))
n_neighbors_mds = 20
for i_x in range(0, n_points):
    for i_y in range(0, n_points):
        if i_x != i_y:
            dif_vect = noisy_sensor_base[:, i_x] - noisy_sensor_base[:, i_y]
            dist_mat_full[i_x, i_y] =  1 / 2 * (numpy.dot(dif_vect.T, numpy.dot(cov_list_full[i_x], dif_vect)) + numpy.dot(dif_vect.T, numpy.dot(cov_list_full[i_y], dif_vect)))
            dist_mat_def[i_x, i_y] =  1 / 2 * (numpy.dot(dif_vect.T, numpy.dot(cov_list_def[i_x], dif_vect)) + numpy.dot(dif_vect.T, numpy.dot(cov_list_def[i_y], dif_vect)))
            dif_vect_real = intrinsic_process_base[:, i_x] - intrinsic_process_base[:, i_y]
            dist_mat_true[i_x, i_y] = numpy.linalg.norm(dif_vect_real)**2
            dist_mat_measured[i_x, i_y] = numpy.linalg.norm(dif_vect)**2

knn_indexes = numpy.argsort(dist_mat_measured, axis=1, kind='quicksort')
knn_indexes = knn_indexes[:, 1:n_neighbors_mds + 1]
isomap_approx = numpy.zeros((n_points, n_points))
for i_x in range(0, n_points):
    for i_y in range(0, n_neighbors_mds):
        isomap_approx[i_x, knn_indexes[i_x, i_y]] = numpy.sqrt(dist_mat_def[i_x, knn_indexes[i_x, i_y]])

knn_indexes = numpy.argsort(dist_mat_true, axis=1, kind='quicksort')
knn_indexes = knn_indexes[:, 1:n_neighbors_mds + 1]
isomap_true = numpy.zeros((n_points, n_points))
for i_x in range(0, n_points):
    for i_y in range(0, n_neighbors_mds):
        isomap_true[i_x, knn_indexes[i_x, i_y]] = numpy.sqrt(dist_mat_true[i_x, knn_indexes[i_x, i_y]])

knn_indexes = numpy.argsort(dist_mat_measured, axis=1, kind='quicksort')
knn_indexes = knn_indexes[:, 1:n_neighbors_mds + 1]
isomap_measured = numpy.zeros((n_points, n_points))
for i_x in range(0, n_points):
    for i_y in range(0, n_neighbors_mds):
        isomap_measured[i_x, knn_indexes[i_x, i_y]] = numpy.sqrt(dist_mat_measured[i_x, knn_indexes[i_x, i_y]])

isomap_approx = scipy.sparse.csgraph.shortest_path(isomap_approx, directed=False)
isomap_true = scipy.sparse.csgraph.shortest_path(isomap_true, directed=False)
isomap_measured = scipy.sparse.csgraph.shortest_path(isomap_measured, directed=False)

# initial_guess_base = isomap_projection.fit_transform(noisy_sensor_base_for_guess.T)

mds = manifold.MDS(n_components=2, max_iter=20000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
pos_1 = mds.fit(isomap_approx).embedding_
pos_2 = mds.fit(isomap_true).embedding_
pos_3 = mds.fit(isomap_measured).embedding_

sigma = numpy.median(isomap_approx)
diff_kernal = numpy.exp(-(isomap_approx ** 2) / (2 * sigma ** 2))
row_sum = numpy.sum(diff_kernal, axis=1)
normlized_kernal = numpy.dot(numpy.diag(1 / row_sum), diff_kernal)
row_sum = numpy.sum(normlized_kernal, axis=1)
U, S, V = numpy.linalg.svd(normlized_kernal)
diff_embedding = U[:, 1:3]

sigma = numpy.median(dist_mat_true)
diff_kernal = numpy.exp(-(dist_mat_true ** 2) / (sigma ** 2))
row_sum = numpy.sum(diff_kernal, axis=1)
normlized_kernal = numpy.dot(numpy.diag(1 / row_sum), diff_kernal)
row_sum = numpy.sum(normlized_kernal, axis=1)
U, S, V = numpy.linalg.svd(normlized_kernal)
diff_embedding_non_int = U[:, 1:3]

print_process(pos_1.T, bounding_shape=None, color_map=color_map[:, :], titleStr="Suggested Intrinsic Isomap", align_points=intrinsic_process_base)
stress, stress_normlized = embbeding_score(intrinsic_process_base, pos_1.T, titleStr="Suggested Intrinsic Isomap")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

print_process(pos_2.T, bounding_shape=None, color_map=color_map[:, :], titleStr="Isomaps using True Intrinsic Distances", align_points=intrinsic_process_base)
stress, stress_normlized = embbeding_score(intrinsic_process_base, pos_2.T, titleStr="Isomaps using True Intrinsic Distances")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

print_process(pos_3.T, bounding_shape=None, color_map=color_map[:, :], titleStr="Isomaps using Measured Distances", align_points=intrinsic_process_base)
stress, stress_normlized = embbeding_score(intrinsic_process_base, pos_3.T, titleStr="Isomaps using Measured Distances")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

print_process(diff_embedding.T, bounding_shape=None, color_map=color_map[:, :], titleStr="Intrinsic Diffusion Maps", align_points=intrinsic_process_base)
stress, stress_normlized = embbeding_score(intrinsic_process_base, diff_embedding.T, titleStr="Intrinsic Diffusion Maps")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

print_process(diff_embedding_non_int.T, bounding_shape=None, color_map=color_map[:, :], titleStr="Diffusion Maps on Intrinsic Space", align_points=intrinsic_process_base)
stress, stress_normlized = embbeding_score(intrinsic_process_base, diff_embedding_non_int.T, titleStr="Diffusion Maps on Intrinsic Space")
print('stress:', stress)
print('stress_normlized:', stress_normlized)

plt.show(block=True)