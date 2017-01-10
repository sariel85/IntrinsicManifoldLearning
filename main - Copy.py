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

#T.config.profile = True


#sim_dir_name = "2D Double Gaussian Potential"
sim_dir_name = "2D Room - Exact Limits"

sim_dir = './' + sim_dir_name

batch_size = 32
patience = 1
max_updates = 100000
min_improvement = 0.01
nodes_drift = 10
nodes_rec = 10
nodes_inv = 26
nodes_gen = 10

n_guess_points = 8000
batch_size_temp = batch_size
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

intrinsic_process_base = numpy.loadtxt(sim_dir + '/' + 'intrinsic_base.txt', delimiter=',', dtype=dtype).T
intrinsic_process_step = numpy.loadtxt(sim_dir + '/' + 'intrinsic_step.txt', delimiter=',', dtype=dtype).T
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy')
intrinsic_variance = intrinsic_variance.astype(dtype=dtype)

dim_intrinsic = intrinsic_process_base.shape[0]

#exact_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_clean_base.txt', delimiter=',')
#exact_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_clean_step.txt', delimiter=',')

noisy_sensor_base = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_base.txt', delimiter=',', dtype=dtype)
noisy_sensor_step = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy_step.txt', delimiter=',', dtype=dtype)

n_points = noisy_sensor_base.shape[1]


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))


#intrinsic_process_base = intrinsic_process_base[:, 0:n_points]
#intrinsic_process_step = intrinsic_process_step[:, 0:n_points]

dim_measurement = noisy_sensor_base.shape[0]
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy')
measurement_variance = measurement_variance.astype(dtype=dtype)


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
noisy_sensor_base_for_guess = noisy_sensor_base[:, points_guess_index]

input_base = noisy_sensor_base
input_step = noisy_sensor_step

if os.path.isfile(sim_dir + '/' + 'initial_guess.npy'):
    initial_guess_base = numpy.load(sim_dir + '/' + 'initial_guess.npy')
    initial_guess_base = initial_guess_base[:, points_guess_index]
elif dim_intrinsic < dim_measurement:
    isomap_projection = Isomap(n_neighbors=n_neighbors, n_components=dim_intrinsic)
    initial_guess_base = isomap_projection.fit_transform(noisy_sensor_base_for_guess.T)
    initial_guess_base = numpy.asarray(initial_guess_base.T, dtype=dtype)
else:
    initial_guess_base = noisy_sensor_base_for_guess


initial_guess_base = min_max_scaler.fit_transform(initial_guess_base.T).T
ax_inv = print_process(intrinsic_process_base, indexs=points_guess_index, bounding_shape=None, color_map=color_map,
                       titleStr="Intrinsic Process")
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

print_process(initial_guess_base, bounding_shape=None, color_map=color_map[points_guess_index, :],
              titleStr="Initial Intrinsic Guess")
plt.savefig(full_dir_name + '/' + 'initial_guess_zoom.png', bbox_inches='tight')
# plt.close()

print_process(initial_guess_base, bounding_shape=None, color_map=color_map[points_guess_index, :],
              titleStr="Initial Intrinsic Guess - Aligned to True Intrinsic Space",
              align_points=intrinsic_process_base[:, points_guess_index], ax_limits=ax_limits_inv)
plt.savefig(full_dir_name + '/' + 'initial_guess_proper_scale.png', bbox_inches='tight')
# plt.close()


if os.path.isfile(sim_dir + '/' + 'save_point.p'):
    ca = pickle.load(open(sim_dir + '/' + 'save_point.p', "rb"))

    rec_output_base_Theano, gen_output_base_init_Theano, gen_output_base_Theano, inv_output_base_init_Theano, inv_output_base_Theano, drift_inv_Theano, cost_rec_pretrain_Theano, cost_gen_pretrain_Theano, cost_gen_Theano, cost_inv_pretrain_Theano, cost_drift_pretrain_Theano, cost_drift_Theano, cost_inv_Theano, cost_intrinsic_Theano, cost_total_Theano = ca.get_cost(dim_intrinsic, intrinsic_variance, measurement_variance, input_base_Theano, input_step_Theano, measurements_base_Theano, initial_guess_base_Theano)

    test_model = theano.function(
        inputs=[input_base_Theano],
        outputs=[rec_output_base_Theano, gen_output_base_Theano, inv_output_base_Theano, drift_inv_Theano],
    )

else:

    #initial_guess_base, err = manifold.locally_linear_embedding(noisy_sensor_base_for_guess.T, n_neighbors=n_neighbors, n_components=dim_intrinsic, method='hessian')


    ca = AutoEncoder(measurements=noisy_sensor_base, n_intrisic_points=n_points, n_input=dim_measurement,
                     dim_intrinsic=intrinsic_process_base.shape[0], n_hidden_rec=nodes_rec, n_hidden_gen=nodes_gen, n_hidden_drift=nodes_drift,
                     n_hidden_inv=nodes_inv, batch_size=batch_size)

    rec_output_base_Theano, gen_output_base_init_Theano, gen_output_base_Theano, inv_output_base_init_Theano, inv_output_base_Theano, drift_inv_Theano, cost_rec_pretrain_Theano, cost_gen_pretrain_Theano, cost_gen_Theano, cost_inv_pretrain_Theano, cost_drift_pretrain_Theano, cost_drift_Theano, cost_inv_Theano, cost_intrinsic_Theano, cost_total_Theano = ca.get_cost(dim_intrinsic, intrinsic_variance, measurement_variance, input_base_Theano, input_step_Theano, measurements_base_Theano, initial_guess_base_Theano)

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

    train = downhill.Dataset([noisy_sensor_base_for_guess.T, initial_guess_base.T], batch_size=20)
    valid = downhill.Dataset([noisy_sensor_base_for_guess.T, initial_guess_base.T], batch_size=0)

    os.makedirs(full_dir_name + '/' + 'pretrain_rec')
    t = time.time()
    for idx, [tm, vm] in enumerate(train_rec_init.iterate(train=train, valid=valid, validate_every=50, patience=3, min_improvement=min_improvement, max_updates=max_updates)):
        if not(idx % 1):
            print('Iteration ', idx+1, ':')
            print('pretrain rec train:', tm['loss'])
            print('pretrain rec valid:', vm['loss'])
            if not(idx % 50):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intermediate Low Dimensional Representation")
                plt.savefig(full_dir_name + '/' + 'pretrain_rec' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
            if vm['loss'] < 5e-5:
                break

    # do stuff
    print(time.time() - t)
    '''
    train_gen_init = downhill.build(
        loss=cost_gen_pretrain_Theano.mean(),
        params=[ca.W_gen_1, ca.b_gen_1, ca.W_gen_2, ca.b_gen_2, ca.W_gen_3, ca.b_gen_3],
        inputs=[initial_guess_base_Theano, measurements_base_Theano],
        algo='rmsprop',
        monitor_gradients=False)

    train = downhill.Dataset((initial_guess_base.T, noisy_sensor_base_for_guess.T), batch_size=batch_size_temp)
    valid = downhill.Dataset((initial_guess_base.T, noisy_sensor_base_for_guess.T), batch_size=0)

    os.makedirs(full_dir_name + '/' + 'pretrain_gen')
    t = time.time()
    for idx, [tm, vm] in enumerate(train_gen_init.iterate(train=train, valid=valid, validate_every=10, patience=3, min_improvement=min_improvement, max_updates=max_updates)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('pretrain gen train:', tm['loss'])
            print('pretrain gen valid:', vm['loss'])
            if not(idx % 100):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Explained Measurement Process", ax_limits=ax_limits_gen)
                plt.savefig(full_dir_name + '/' + 'pretrain_gen' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
            if vm['loss'] < 5e-3:
                break

    train_inv_init = downhill.build(
        loss=cost_inv_pretrain_Theano.mean(),
        params=[ca.W_inv_1, ca.b_inv_1, ca.W_inv_2, ca.b_inv_2, ca.W_inv_3, ca.b_inv_3],
        inputs=[initial_guess_base_Theano],
        algo='rmsprop',
        monitor_gradients=False)

    train = downhill.Dataset([initial_guess_base.T], batch_size=batch_size_temp)
    valid = downhill.Dataset([initial_guess_base.T], batch_size=0)

    os.makedirs(full_dir_name + '/' + 'pretrain_inv')
    for idx, [tm, vm] in enumerate(train_inv_init.iterate(train=train, valid=valid, validate_every=10, patience=3, min_improvement=min_improvement, max_updates=max_updates)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('pretrain inv train:', tm['loss'])
            print('pretrain inv valid:', vm['loss'])
            if not(idx % 100):
                    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                    print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Reconstructed Intrinsic Process")
                    plt.savefig(full_dir_name + '/' + 'pretrain_inv' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()
            if vm['loss'] < 1e-4:
                break
    '''
    '''
    train_gen = downhill.build(
        loss=cost_gen_Theano.mean(),
        params=[ca.W_gen_1, ca.b_gen_1, ca.W_gen_2, ca.b_gen_2, ca.W_gen_3, ca.b_gen_3],
        inputs=[input_base_Theano, measurements_base_Theano],
        algo='rmsprop',
        monitor_gradients=False)

    train = downhill.Dataset([input_base.T, noisy_sensor_base.T], batch_size=batch_size_temp)
    valid = downhill.Dataset([input_base.T, noisy_sensor_base.T], batch_size=1000)

    os.makedirs(full_dir_name + '/' + 'gen')
    for idx, [tm, vm] in enumerate(train_gen.iterate(train=train, valid=valid, learning_rate=1e-4, validate_every=10, patience=1, min_improvement=min_improvement, max_updates=max_updates)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('f loss train:', tm['loss'])
            print('f loss valid:', vm['loss'])
            if not(idx % 10):
                if not (idx == 0):
                    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                    print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Reconstructed Intrinsic Process")
                    plt.savefig(full_dir_name + '/' + 'gen' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()
                    print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intermediate Low Dimensional Representation")
                    plt.savefig(full_dir_name + '/' + 'gen' + '/' + 'rec_' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()
                    print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Explained Measurement Process", ax_limits=ax_limits_gen)
                    plt.savefig(full_dir_name + '/' + 'gen' + '/' + 'gen_' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()
    '''
    '''
    pretrain_drift = downhill.build(
        loss=cost_drift_pretrain_Theano,
        params=[ca.W_drift_1, ca.b_drift_1, ca.W_drift_2, ca.b_drift_2, ca.W_drift_3, ca.b_drift_3],
        inputs=[input_base_Theano],
        algo='rmsprop',
        monitor_gradients=False)

    train = downhill.Dataset([input_base.T], batch_size=batch_size)
    valid = downhill.Dataset([input_base.T], batch_size=n_guess_points)

    os.makedirs(full_dir_name + '/' + 'pretrain_drift')
    for idx, [tm, vm] in enumerate(pretrain_drift.iterate(train=train, valid=valid, validate_every=10, patience=3, min_improvement=min_improvement, max_updates=max_updates)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('pretrain_drift loss train:', tm['loss'])
            print('pretrain_drift loss valid:', vm['loss'])
            if not(idx % 10):
                if not(idx == 0):
                    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                    print_drift(inv_output_base_val, drift_inv_val, titleStr='Drift', color_map=color_map[points_plot_index, :])
                    plt.savefig(full_dir_name + '/' + 'pretrain_drift' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()
            if vm['loss'] < 5e-3:
                break


    train_drift = downhill.build(
        loss=cost_drift_Theano,
        params=[ca.W_drift_1, ca.b_drift_1, ca.W_drift_2, ca.b_drift_2, ca.W_drift_3, ca.b_drift_3],
        inputs=[input_base_Theano, input_step_Theano],
        algo='rmsprop',
        monitor_gradients=False)

    train = downhill.Dataset([input_base.T, input_step.T], batch_size=batch_size_temp)
    valid = downhill.Dataset([input_base.T, input_step.T], batch_size=0)

    os.makedirs(full_dir_name + '/' + 'drift')
    for idx, [tm, vm] in enumerate(train_drift.iterate(train=train, valid=valid, validate_every=1, patience=3, min_improvement=0.01*min_improvement, max_updates=max_updates)):
        if not(idx % 1):
            print('Iteration ', idx + 1, ':')
            print('drift loss train:', tm['loss'])
            print('drift loss valid:', vm['loss'])
            if not(idx % 1):
                if not(idx == 0):
                    [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                    print_drift(inv_output_base_val, drift_inv_val, titleStr='Drift', color_map=color_map[points_plot_index, :])
                    plt.savefig(full_dir_name + '/' + 'drift' + '/' + format(round(time.time())) + '.png', bbox_inches='tight')
                    plt.close()
    '''
    f = open(sim_dir + '/' + 'save_point.p', 'wb')
    pickle.dump(ca, f)
    f.close()


train_inv = downhill.build(
    loss=cost_inv_Theano,
    params=[ca.W_inv_1, ca.b_inv_1, ca.W_inv_2, ca.b_inv_2, ca.W_inv_3, ca.b_inv_3],
    inputs=[input_base_Theano, input_step_Theano],
    algo='rmsprop',
    monitor_gradients=False)

train = downhill.Dataset([input_base.T, input_step.T], batch_size=20)
valid = downhill.Dataset([input_base.T, input_step.T], batch_size=0)

os.makedirs(full_dir_name + '/' + 'inv')
cost_term = []
stress_term = []
for idx, [tm, vm] in enumerate(train_inv.iterate(train=train, valid=valid, validate_every=50, patience=5, min_improvement=min_improvement, max_updates=1000)):
    if not(idx % 10):
        print('Iteration ', idx + 1, ':')
        print('inv loss train:', tm['loss'])
        print('inv loss valid:', vm['loss'])
        if not(idx % 50):
                [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
                print_drift(inv_output_base_val, drift_inv_val, titleStr='Estimated Intrinsic Representation and Drift', color_map=color_map[points_plot_index, :])
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'drift_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Estimated Intrinsic Representation - Aligned to True Intrinsic Space", ax_limits=ax_limits_inv, align_points=intrinsic_process_base[:, points_plot_index])
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intermediate Low Dimensional Representation")
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'rec_' + format(round(time.time())) + '.png', bbox_inches='tight')
                print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intermediate Low Dimensional Representation")
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'gen_' + format(round(time.time())) + '.png', bbox_inches='tight')
                stress, stress_normlized = embbeding_score(intrinsic_process_base[:, points_plot_index], inv_output_base_val)
                print('stress:', stress)
                print('stress_normlized:', stress_normlized)
                cost_term.append(vm['loss'])
                stress_term.append(stress_normlized)
                plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'stress_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cost_term[1:], c='b')
ax.set_xlabel('Time [Iterations]')
ax.set_ylabel('Cost')
ax.set_title("Cost Minimization")
plt.show(block=False)
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'prog_cost.png', bbox_inches='tight')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(stress_term[1:], c='r')
ax.set_xlabel('Time [Iterations]')
ax.set_ylabel('Stress')
ax.set_title("Embedding Stress")
plt.show(block=False)
plt.savefig(full_dir_name + '/' + 'inv' + '/' + 'prog_stress.png', bbox_inches='tight')
plt.close()

train_total = downhill.build(
    loss=cost_total_Theano,
    params=[ca.W_rec_1, ca.b_rec_1, ca.W_rec_2, ca.b_rec_2, ca.W_rec_3, ca.b_rec_3, ca.W_gen_1, ca.b_gen_1, ca.W_gen_2, ca.b_gen_2, ca.W_gen_3, ca.b_gen_3, ca.W_inv_1, ca.b_inv_1, ca.W_inv_2,
            ca.b_inv_2, ca.W_inv_3],
    inputs=[input_base_Theano, input_step_Theano, measurements_base_Theano],
    algo='rmsprop',
    monitor_gradients=False)

train = downhill.Dataset([input_base.T, input_step.T, noisy_sensor_base.T], batch_size=batch_size_temp)
valid = downhill.Dataset([input_base.T, input_step.T, noisy_sensor_base.T], batch_size=0)

os.makedirs(full_dir_name + '/' + 'total')
cost_term = []
stress_term = []
for idx, [tm, vm] in enumerate(train_total.iterate(train=train, valid=valid, validate_every=10, patience=3, min_improvement=min_improvement, max_updates=300)):
    if not(idx % 1):
        print('Iteration ', idx + 1, ':')
        print('total loss train:', tm['loss'])
        print('total loss valid:', vm['loss'])
        [rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
        stress, stress_normlized = embbeding_score(intrinsic_process_base[:, points_plot_index], inv_output_base_val)
        print('stress:', stress)
        print('stress_normlized:', stress_normlized)
        cost_term.append(vm['loss'])
        stress_term.append(stress_normlized)
        if not(idx % 10):
                print_drift(inv_output_base_val, drift_inv_val, titleStr='Reconstructed Intrinsic Representation and Drift', color_map=color_map[points_plot_index, :])
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'drift_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Intrinsic Representation Updates", ax_limits=ax_limits_inv, align_points=intrinsic_process_base[:, points_plot_index])
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'inv_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(gen_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Messurment Updates", ax_limits=ax_limits_gen)
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'gen_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                print_process(rec_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :], titleStr="Messurment Updates", ax_limits=ax_limits_gen)
                plt.savefig(full_dir_name + '/' + 'total' + '/' + 'rec_' + format(round(time.time())) + '.png', bbox_inches='tight')
                plt.close()
                stress, stress_normlized = embbeding_score(intrinsic_process_base[:, points_plot_index], inv_output_base_val)
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


[rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base[:, points_plot_index].T)
print_drift(inv_output_base_val, drift_inv_val, titleStr='Reconstructed Intrinsic Representation and Drift',
            color_map=color_map[points_plot_index, :])
plt.savefig(full_dir_name + '/' + format(round(time.time())) + '_' + 'drift.png', bbox_inches='tight')
print_process(inv_output_base_val, bounding_shape=None, color_map=color_map[points_plot_index, :],
              titleStr="Intrinsic Representation Updates")
plt.savefig(full_dir_name + '/' + format(round(time.time())) + '_' + 'inv.png', bbox_inches='tight')

[rec_output_base_val, gen_output_base_val, inv_output_base_val, drift_inv_val] = test_model(input_base.T)

'''
f = open(sim_dir + '/' + 'net.save', 'wb')
pickle.dump(ca, f)
f.close()
'''
#numpy.save(sim_dir + '/' + 'initial_guess.npy', inv_output_base_val)
