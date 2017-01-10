from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import print_process, print_drift, create_color_map
import time
import downhill
import os
import theano.tensor as T
import theano.tensor.nlinalg
from Autoencoder import AutoEncoder
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from Util import *
import pickle
from sklearn import preprocessing
from sklearn import manifold
from diff_maps import diff_maps


#sim_dir_name = "2D Double Gaussian Potential"
sim_dir_name = "Non Tangent"

sim_dir = './' + sim_dir_name

batch_size = 32
patience = 1
max_updates = 150000
min_improvement = 0.01
nodes_drift = 20
nodes_rec = 20
nodes_inv = 30
nodes_gen = 10

n_guess_points = 100
iteration_size_temp = 1
precision = 'float64'

if precision == 'float32':
    dtype = numpy.float32
    input_base_Theano = T.fmatrix('input_base_Theano')
    input_step_Theano = T.fmatrix('input_step_Theano')
    measurements_base_Theano = T.fmatrix('measurements_base_Theano')
    initial_guess_base_Theano = T.fmatrix('initial_guess_base_Theano')

elif precision == 'float64':
    dtype = numpy.float64
    input_base_Theano = T.dmatrix('input_base_Theano')
    input_step_Theano = T.dmatrix('input_step_Theano')
    measurements_base_Theano = T.dmatrix('measurements_base_Theano')
    initial_guess_base_Theano = T.dmatrix('initial_guess_base_Theano')

theano.config.floatX = precision


#Initial Guess Properties
n_neighbors = 20

#Visualization properties
n_plot_points = 2000
#Create unique folder based on timestamp to save figures in
ts = time.time()
ts = round(ts)
dir_name = '/{}'.format(ts)
full_dir_name = './' + 'Runs' + '/' + sim_dir_name + dir_name
os.makedirs(full_dir_name)

intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',', dtype=dtype)
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',', dtype=dtype)
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy')
intrinsic_variance = intrinsic_variance.astype(dtype=dtype)

dim_intrinsic = intrinsic_process_base.shape[0]

#exact_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_clean_base.txt', delimiter=',')
#exact_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_clean_step.txt', delimiter=',')

noisy_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_base.txt', delimiter=',', dtype=dtype)
noisy_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_step.txt', delimiter=',', dtype=dtype)

n_points = noisy_sensor_base.shape[1]


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-0.5, 0.5))


#intrinsic_process_base = intrinsic_process_base[:, 0:n_points]
#intrinsic_process_step = intrinsic_process_step[:, 0:n_points]

dim_measurement = noisy_sensor_base.shape[0]
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy')
measurement_variance = measurement_variance.astype(dtype=dtype)

measurement_variance = 0.001
#############################################################################################

n_plot_points = min(n_points, n_plot_points)
points_plot_index = numpy.random.choice(n_points, size=n_plot_points, replace=False)

#############################################################################################

# Generate point coloring based on intrinsic coordinates
color_map = create_color_map(intrinsic_process_base)

#############################################################################################

# Get Low Dimensional Starting Point
n_guess_points = min(n_points, n_guess_points)
points_guess_index = numpy.random.choice(n_points, size=n_guess_points, replace=False)


scale = numpy.max(noisy_sensor_base)-numpy.min(noisy_sensor_base)
origin = numpy.mean(noisy_sensor_base, axis=1)
input_base = (noisy_sensor_base.T-origin.T).T/scale
input_step = (noisy_sensor_step.T-origin.T).T/scale

#input_base = min_max_scaler.fit_transform(noisy_sensor_base.T).T
#input_step = min_max_scaler.fit_transform(noisy_sensor_step.T).T

#noisy_sensor_base_for_guess = noisy_sensor_base[:, points_guess_index]

#temp = min_max_scaler.fit_transform([noisy_sensor_base.T,noisy_sensor_step.T]).T

#if os.path.isfile(sim_dir + '/' + 'initial_guess.npy'):
#    initial_guess_base = numpy.load(sim_dir + '/' + 'initial_guess.npy')
#    initial_guess_base = initial_guess_base[:, points_guess_index]
#else:
#    #isomap_projection = Isomap(n_neighbors=n_neighbors, n_components=dim_intrinsic)
#    #initial_guess_base = isomap_projection.fit_transform(noisy_sensor_base_for_guess.T).T
#    initial_guess_base = manifold.LocallyLinearEmbedding(n_neighbors, dim_intrinsic, eigen_solver='auto', method='modified').fit_transform(noisy_sensor_base_for_guess.T).T

#else:

#initial_guess_base_scaled = input_base[:, points_guess_index]
#initial_guess_base = noisy_sensor_base[:, points_guess_index]

noisy_sensor_base_for_guess = input_base[:, points_guess_index]

ax_inv = print_process(intrinsic_process_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map,titleStr="Intrinsic Process")
plt.savefig(full_dir_name + '/' + 'intrinsic_base.png', bbox_inches='tight')
# plt.close()
# print_process(exact_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Noiseless Sensor Process")
ax_gen = print_process(noisy_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map,
                       titleStr="Noisy Sensor Process")
plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')

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
    ax_limits_gen = None

pos_1, pos_2, pos_3, diff_embedding, diff_embedding_non_int = diff_maps(points_cluster_index=points_guess_index, noisy_sensor_base=noisy_sensor_base, noisy_sensor_step=noisy_sensor_step, intrinsic_variance=intrinsic_variance, intrinsic_process_base=intrinsic_process_base, intrinsic_process_step=intrinsic_process_step, dim_intrinsic=2, n_neighbors_cov = 50, n_neighbors_mds = 20, ax_limits_inv=ax_limits_inv, ax_limits_gen=ax_limits_gen)

initial_guess_base = pos_1.T

initial_guess_base_scaled = min_max_scaler.fit_transform(initial_guess_base.T).T
#noisy_sensor_base_for_guess = min_max_scaler.fit_transform(noisy_sensor_base_for_guess.T).T

print_process(initial_guess_base, bounding_shape=None, color_map=color_map[points_guess_index, :],
              titleStr="Initial Embedding")
plt.savefig(full_dir_name + '/' + 'initial_guess_zoom.png', bbox_inches='tight')
# plt.close()

print_process(initial_guess_base, bounding_shape=None, color_map=color_map[points_guess_index, :],
              titleStr="Initial Embedding",
              align_points=intrinsic_process_base[:, points_guess_index], ax_limits=ax_limits_inv)
plt.savefig(full_dir_name + '/' + 'initial_guess_proper_scale.png', bbox_inches='tight')
# plt.close()


if os.path.isfile(sim_dir + '/' + 'save_point.p'):
    ca = pickle.load(open(sim_dir + '/' + 'save_point.p', "rb"))

    rec_output_base_Theano, gen_output_base_init_Theano, gen_output_base_Theano, inv_output_base_init_Theano, inv_output_base_Theano, drift_inv_Theano, cost_rec_pretrain_Theano, cost_gen_pretrain_Theano, cost_gen_Theano, cost_inv_pretrain_Theano, cost_drift_pretrain_Theano, cost_drift_Theano, cost_inv_Theano, cost_intrinsic_Theano, cost_measurements_reg_Theano, cost_total_Theano = ca.get_cost(dim_intrinsic, intrinsic_variance, measurement_variance, input_base_Theano, input_step_Theano, measurements_base_Theano, initial_guess_base_Theano)

    test_model = theano.function(
        inputs=[input_base_Theano],
        outputs=[rec_output_base_Theano, gen_output_base_Theano, inv_output_base_Theano, drift_inv_Theano],
    )

    os.makedirs(full_dir_name + '/' + 'drift')


else:

    #initial_guess_base, err = manifold.locally_linear_embedding(noisy_sensor_base_for_guess.T, n_neighbors=n_neighbors, n_components=dim_intrinsic, method='hessian')


    ca = AutoEncoder(measurements=noisy_sensor_base, n_intrisic_points=n_points, dim_input=dim_measurement,
                     dim_intrinsic=intrinsic_process_base.shape[0], n_hidden_rec=nodes_rec, n_hidden_gen=nodes_gen, n_hidden_drift=nodes_drift,
                         n_hidden_inv=nodes_inv, batch_size=batch_size)

    rec_output_base_Theano, gen_output_base_init_Theano, gen_output_base_Theano, inv_output_base_init_Theano, inv_output_base_Theano, drift_inv_Theano, cost_rec_pretrain_Theano, cost_gen_pretrain_Theano, cost_gen_Theano, cost_inv_pretrain_Theano, cost_drift_pretrain_Theano, cost_drift_Theano, cost_inv_Theano, cost_intrinsic_Theano, cost_measurements_reg_Theano, cost_total_Theano = ca.get_cost(dim_intrinsic, intrinsic_variance, measurement_variance, input_base_Theano, input_step_Theano, measurements_base_Theano, initial_guess_base_Theano)

    test_model = theano.function(
        inputs=[input_base_Theano],
        outputs=[rec_output_base_Theano, gen_output_base_Theano, inv_output_base_Theano, drift_inv_Theano],
    )

    train_rec_init = downhill.build(
        loss=cost_rec_pretrain_Theano.mean(),
        params=[ca.W_rec_1, ca.b_rec_1, ca.W_rec_2, ca.b_rec_2, ca.W_rec_3, ca.b_rec_3],
        inputs=[input_base_Theano, initial_guess_base_Theano],
        algo='rmsprop',
        monitor_gradients=False)

    train = downhill.Dataset([noisy_sensor_base_for_guess.T, initial_guess_base_scaled.T], batch_size=100)
    valid = downhill.Dataset([noisy_sensor_base_for_guess.T, initial_guess_base_scaled.T], batch_size=0)

    os.makedirs(full_dir_name + '/' + 'pretrain_rec')
    t = time.time()
    for idx, [tm, vm] in enumerate(train_rec_init.iterate(train=train, valid=valid, validate_every=100, patience=100, min_improvement=min_improvement, max_updates=2001)):
        if not(idx % 1):
            print('Iteration ', idx+1, ':')
            print('pretrain rec train:', tm['loss'])
            print('pretrain rec valid:', vm['loss'])
            if not(idx % 100):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(noisy_sensor_base_for_guess.T)
                print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_guess_index, :], titleStr="Intermediate Low Dimensional Representation")
                plt.savefig(full_dir_name + '/' + 'pretrain_rec' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
            if vm['loss'] < 5e-5:
                break

    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(noisy_sensor_base_for_guess.T)
    print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_guess_index, :],
                  titleStr="Final Init Rec")
    plt.savefig(full_dir_name + '/' + 'pretrain_rec' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
    plt.close()

    '''
    train_gen_init = downhill.build(
        loss=cost_gen_pretrain_Theano,
        params=[ca.W_gen_1, ca.b_gen_1, ca.W_gen_2, ca.b_gen_2, ca.W_gen_3, ca.b_gen_3],
        inputs=[initial_guess_base_Theano, measurements_base_Theano],
        algo='adagrad',
        monitor_gradients=False)

    train = downhill.Dataset((initial_guess_base.T, noisy_sensor_base_for_guess.T), batch_size=batch_size_temp)
    valid = downhill.Dataset((initial_guess_base.T, noisy_sensor_base_for_guess.T), batch_size=0)

    os.makedirs(full_dir_name + '/' + 'pretrain_gen')
    t = time.time()
    for idx, [tm, vm] in enumerate(train_gen_init.iterate(train=train, valid=valid, learning_rate=1e-1, validate_every=100, patience=3, min_improvement=min_improvement, max_updates=max_updates)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('pretrain gen train:', tm['loss'])
            print('pretrain gen valid:', vm['loss'])
            if not(idx % 200):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(noisy_sensor_base_for_guess.T)
                print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_guess_index, :], titleStr="Explained Measurement Process", ax_limits=ax_limits_gen)
                plt.savefig(full_dir_name + '/' + 'pretrain_gen' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
            if vm['loss'] < 5e-4:
                break
    '''

    train_inv_init = downhill.build(
        loss=cost_inv_pretrain_Theano.mean(),
        params=[ca.W_inv_1, ca.b_inv_1, ca.W_inv_2, ca.b_inv_2, ca.W_inv_3, ca.b_inv_3],
        inputs=[initial_guess_base_Theano],
        algo='rmsprop',
        monitor_gradients=False)

    os.makedirs(full_dir_name + '/' + 'pretrain_inv')


    [rec_output_init, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base.T)

    print_process(rec_output_init, bounding_shape=None, color_map=color_map,
                  titleStr="Reconstructed Intrinsic Process")

    train = downhill.Dataset([rec_output_init.T], batch_size=1000)
    valid = downhill.Dataset([rec_output_init.T], batch_size=0)

    for idx, [tm, vm] in enumerate(train_inv_init.iterate(train=train, valid=valid, validate_every=100, patience=100, min_improvement=min_improvement, max_updates=4001)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('pretrain inv train:', tm['loss'])
            print('pretrain inv valid:', vm['loss'])
            if not(idx % 100):
                    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                    print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Reconstructed Intrinsic Process")
                    plt.savefig(full_dir_name + '/' + 'pretrain_inv' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()
            if vm['loss'] < 5e-5:
                break

    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
    print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
                  titleStr="Final Inv Init")
    plt.savefig(full_dir_name + '/' + 'pretrain_inv' + '/' + 'inv_' + format(round(time.time())) + '.png',
                bbox_inches='tight')
    plt.close()

    pretrain_drift = downhill.build(
        loss=cost_drift_pretrain_Theano.mean(),
        params=[ca.W_drift_1, ca.b_drift_1, ca.W_drift_2, ca.b_drift_2, ca.W_drift_3, ca.b_drift_3],
        inputs=[input_base_Theano, input_step_Theano],
        algo='adadelta',
        monitor_gradients=False)

    train = downhill.Dataset([input_base.T, input_step.T], batch_size=100, iteration_size=100)
    valid = downhill.Dataset([input_base.T, input_step.T], batch_size=0)

    os.makedirs(full_dir_name + '/' + 'pretrain_drift')
    for idx, [tm, vm] in enumerate(pretrain_drift.iterate(train=train, valid=valid, validate_every=100, patience=100, min_improvement=0.001, max_updates=1001)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('pretrain_drift loss train:', tm['loss'])
            print('pretrain_drift loss valid:', vm['loss'])
            if not(idx % 100):
                if not (idx == 0):
                    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                    print_drift(inv_output_base_val, drift_inv_val, titleStr='Drift')
                    plt.savefig(full_dir_name + '/' + 'pretrain_drift' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()
            if vm['loss'] < 1e-5:
                break

    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(
        input_base[:, points_plot_index].T)
    print_drift(inv_output_base_val, drift_inv_val, titleStr='Final Pretrain Drift', color_map=color_map[points_plot_index, :])
    plt.savefig(full_dir_name + '/' + 'pretrain_drift' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
    plt.close()

    train_drift = downhill.build(
        loss=cost_drift_Theano.mean(),
        params=[ca.W_drift_1, ca.b_drift_1, ca.W_drift_2, ca.b_drift_2, ca.W_drift_3, ca.b_drift_3],
        inputs=[input_base_Theano, input_step_Theano],
        algo='rmsprop',
        monitor_gradients=False)

    train = downhill.Dataset([input_base.T, input_step.T], batch_size=100, iteration_size=100)
    valid = downhill.Dataset([input_base.T, input_step.T], batch_size=0)

    os.makedirs(full_dir_name + '/' + 'drift')
    for idx, [tm, vm] in enumerate(train_drift.iterate(train=train, valid=valid, learning_rate=1e-5, validate_every=100, patience=3, min_improvement=0.001, max_updates=1001)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('drift loss train:', tm['loss'])
            print('drift loss valid:', vm['loss'])
            if not(idx % 100):
                if not(idx == 0):
                    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                    print_drift(inv_output_base_val, drift_inv_val, titleStr='Drift')
                    plt.savefig(full_dir_name + '/' + 'drift' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()

    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(
        input_base[:, points_plot_index].T)
    print_drift(inv_output_base_val, drift_inv_val, titleStr='Final Drift Estimation')
    plt.savefig(full_dir_name + '/' + 'drift' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
    plt.close()

    f = open(sim_dir + '/' + 'save_point.p', 'wb')
    pickle.dump(ca, f)
    f.close()

train_inv = downhill.build(
    loss=cost_inv_Theano.mean(),
    params=[ca.W_inv_1, ca.b_inv_1, ca.W_inv_2, ca.b_inv_2, ca.W_inv_3, ca.b_inv_3],
    inputs=[input_base_Theano, input_step_Theano],
    algo='adam',
    monitor_gradients=False)

train = downhill.Dataset([input_base.T, input_step.T], batch_size=batch_size, iteration_size=50)
valid = downhill.Dataset([input_base.T, input_step.T], batch_size=0)

os.makedirs(full_dir_name + '/' + 'inv')
cost_term = []
stress_term = []

[rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(
    input_base[:, points_plot_index].T)
print_drift(inv_output_base_val, drift_inv_val, titleStr='Estimated Intrinsic Representation and Drift')
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'drift_' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()
print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
              titleStr="Estimated Intrinsic Representation", ax_limits=ax_limits_inv,
              align_points=intrinsic_process_base[:, points_plot_index])
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()
print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
              titleStr="Intermediate Low Dimensional Representation")
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'rec_' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()
print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
              titleStr="Intermediate Low Dimensional Representation")
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'gen_' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()

for idx, [tm, vm] in enumerate(train_inv.iterate(train=train, valid=valid, learning_rate=0.001, validate_every=100, patience=100, min_improvement=min_improvement, max_updates=4001)):
    if not(idx % 1):
        print('Iteration ', idx + 1, ':')
        print('inv loss train:', tm['loss'])
        print('inv loss valid:', vm['loss'])
        if not(idx % 100):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                print_drift(inv_output_base_val, drift_inv_val, titleStr='Estimated Intrinsic Representation and Drift')
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'drift_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Estimated Intrinsic Representation", ax_limits=ax_limits_inv, align_points=intrinsic_process_base[:, points_plot_index])
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                #print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intermediate Low Dimensional Representation")
                #plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'rec_' + format(round(time.time())) + '.png', bbox_inches='tight')
                #plt.close()
                #print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intermediate Low Dimensional Representation")
                #plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'gen_' + format(round(time.time())) + '.png', bbox_inches='tight')
                #plt.close()
                stress, stress_normlized = embbeding_score(intrinsic_process_base[:, points_plot_index], inv_output_base_val, titleStr="Embedding Distance Estimation")
                print('stress:', stress)
                print('stress_normalized:', stress_normlized)
                cost_term.append(vm['loss'])
                stress_term.append(stress_normlized)
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'stress_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()


train_drift = downhill.build(
    loss=cost_drift_Theano.mean(),
    params=[ca.W_drift_1, ca.b_drift_1, ca.W_drift_2, ca.b_drift_2, ca.W_drift_3, ca.b_drift_3],
    inputs=[input_base_Theano, input_step_Theano],
    algo='rmsprop',
    monitor_gradients=False)

train = downhill.Dataset([input_base.T, input_step.T], batch_size=100)
valid = downhill.Dataset([input_base.T, input_step.T], batch_size=0)

for idx, [tm, vm] in enumerate(
        train_drift.iterate(train=train, valid=valid, learning_rate=1e-5, validate_every=100, patience=3, min_improvement=0.01,
                            max_updates=1001)):
    if not (idx % 1):
        print('Iteration ', idx + 1, ':')
        print('drift loss train:', tm['loss'])
        print('drift loss valid:', vm['loss'])
        if not (idx % 100):
            if not (idx == 0):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(
                    input_base[:, points_plot_index].T)
                print_drift(inv_output_base_val, drift_inv_val, titleStr='Drift')
                plt.savefig(full_dir_name + '/' + 'drift' + '/' + format(round(time.time())) + '.png',
                            bbox_inches='tight')
                plt.close()

[rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(
    input_base[:, points_plot_index].T)
print_drift(inv_output_base_val, drift_inv_val, titleStr='Final Drift Estimation')
plt.savefig(full_dir_name + '/' + 'drift' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()

train_inv = downhill.build(
    loss=cost_inv_Theano.mean(),
    params=[ca.W_inv_1, ca.b_inv_1, ca.W_inv_2, ca.b_inv_2, ca.W_inv_3, ca.b_inv_3],
    inputs=[input_base_Theano, input_step_Theano],
    algo='adam',
    monitor_gradients=False)

train = downhill.Dataset([input_base.T, input_step.T], batch_size=batch_size, iteration_size=50)
valid = downhill.Dataset([input_base.T, input_step.T], batch_size=0)

[rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(
    input_base[:, points_plot_index].T)
print_drift(inv_output_base_val, drift_inv_val, titleStr='Estimated Intrinsic Representation and Drift')
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'drift_' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()
print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
              titleStr="Estimated Intrinsic Representation", ax_limits=ax_limits_inv,
              align_points=intrinsic_process_base[:, points_plot_index])
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()
print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
              titleStr="Intermediate Low Dimensional Representation")
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'rec_' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()
print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
              titleStr="Intermediate Low Dimensional Representation")
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'gen_' + format(round(time.time())) + '.png', bbox_inches='tight')
plt.close()

for idx, [tm, vm] in enumerate(train_inv.iterate(train=train, valid=valid, learning_rate=0.0005, validate_every=100, patience=100, min_improvement=min_improvement, max_updates=2001)):
    if not(idx % 1):
        print('Iteration ', idx + 1, ':')
        print('inv loss train:', tm['loss'])
        print('inv loss valid:', vm['loss'])
        if not(idx % 100):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                print_drift(inv_output_base_val, drift_inv_val, titleStr='Estimated Intrinsic Representation and Drift')
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'drift_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Estimated Intrinsic Representation", ax_limits=ax_limits_inv, align_points=intrinsic_process_base[:, points_plot_index])
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                #print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intermediate Low Dimensional Representation")
                #plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'rec_' + format(round(time.time())) + '.png', bbox_inches='tight')
                #plt.close()
                #print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intermediate Low Dimensional Representation")
                #plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'gen_' + format(round(time.time())) + '.png', bbox_inches='tight')
                #plt.close()
                stress, stress_normlized = embbeding_score(intrinsic_process_base[:, points_plot_index], inv_output_base_val, titleStr="Embedding Distance Estimation")
                print('stress:', stress)
                print('stress_normalized:', stress_normlized)
                cost_term.append(vm['loss'])
                stress_term.append(stress_normlized)
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'stress_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cost_term[1:], c='b')
ax.set_xlabel('Epochs')
ax.set_ylabel('Cost')
ax.set_title("Cost vs Epochs")
plt.show(block=False)
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'prog_cost.png', bbox_inches='tight')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(stress_term[1:], c='r')
ax.set_xlabel('Epochs')
ax.set_ylabel('Stress')
ax.set_title("Embedding Stress vs Epochs")
plt.show(block=False)
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'prog_stress.png', bbox_inches='tight')
plt.close()

'''
train_total = downhill.build(
    loss=cost_total_Theano,
    params=[ca.W_rec_1, ca.b_rec_1, ca.W_rec_2, ca.b_rec_2, ca.W_rec_3, ca.b_rec_3, ca.W_gen_1, ca.b_gen_1, ca.W_gen_2, ca.b_gen_2, ca.W_gen_3, ca.b_gen_3, ca.W_inv_1, ca.b_inv_1, ca.W_inv_2,
            ca.b_inv_2, ca.W_inv_3],
    inputs=[input_base_Theano, input_step_Theano, measurements_base_Theano],
    algo='adagrad',
    monitor_gradients=False)

train = downhill.Dataset([input_base.T, input_step.T, noisy_sensor_base.T], batch_size=batch_size_temp)
valid = downhill.Dataset([input_base.T, input_step.T, noisy_sensor_base.T], batch_size=0)

os.makedirs(full_dir_name + '/' + 'total')
cost_term = []
stress_term = []
for idx, [tm, vm] in enumerate(train_total.iterate(train=train, valid=valid, momentum=0.9, learning_rate=1e-2, validate_every=10, patience=3, min_improvement=min_improvement, max_updates=300)):
    if not(idx % 1):
        print('Iteration ', idx + 1, ':')
        print('total loss train:', tm['loss'])
        print('total loss valid:', vm['loss'])
        cost_term.append(vm['loss'])
        stress_term.append(stress_normlized)
        if not(idx % 10):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                print_drift(inv_output_base_val, drift_inv_val, titleStr='Reconstructed Intrinsic Representation and Drift', color_map=color_map[points_plot_index, :])
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'drift_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intrinsic Representation Updates", ax_limits=ax_limits_inv, align_points=intrinsic_process_base[:, points_plot_index])
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Measurement Updates", ax_limits=ax_limits_gen)
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'gen_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Measurement Updates")
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'rec_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                stress, stress_normalized = embbeding_score(intrinsic_process_base[:, points_plot_index], inv_output_base_val)
                print('stress:', stress)
                print('stress_normalized:', stress_normlized)
                cost_term.append(vm['loss'])
                stress_term.append(stress_normlized)
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'stress_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cost_term[1:], c='b')
ax.set_xlabel('Time [Iterations]')
ax.set_ylabel('Cost')
ax.set_title("Cost Minimization")
plt.show(block=False)
plt.savefig(full_dir_name + '/' + 'total' + '/' + 'prog_cost.png', bbox_inches='tight')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(stress_term[1:], c='r')
ax.set_xlabel('Time [Iterations]')
ax.set_ylabel('Stress')
ax.set_title("Embedding Stress")
plt.show(block=False)
plt.savefig(full_dir_name + '/' + 'total' + '/' + 'prog_stress.png', bbox_inches='tight')
plt.close()
'''

[rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
              titleStr="Intrinsic Representation Updates", ax_limits=ax_limits_inv, align_points=intrinsic_process_base[:, points_plot_index])
plt.savefig(full_dir_name + '/' + 'final_inv.png', bbox_inches='tight')


[rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base.T)

'''
f = open(sim_dir + '/' + 'net.save', 'wb')
pickle.dump(ca, f)
f.close()
'''

numpy.save(sim_dir + '/' + 'initial_guess.npy', inv_output_base_val)
