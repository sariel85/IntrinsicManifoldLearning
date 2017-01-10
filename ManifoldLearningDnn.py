from __future__ import print_function
import keras
import theano.tensor as K
import numpy
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.layers.core import *
from DataGeneration import print_process
from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier

def one_level(sampled_process, intrinsic_process, patch, color_map, process_var, bounding_shape):
    dim_intrinsinc = intrinsic_process.shape[0]
    per_dist_approx = 0.9
    n_recon_points = 4000
    n_epoch_recon = 4000
    n_cluster_points_per_patch = 400
    n_knn_cov_est = 100
    n_knn_cov_est_assessment = 300
    n_epoch_patch_rectifier = 5000
    n_max_pairs_for_patch_rectification_assessment = 3000
    n_max_pairs_from_patch_for_global = 2000
    n_dist_used_manifold_unfolder = 200
    n_epoch_manifold_unfolder = 10000
    approx_method = "Mid-Point"
    n_repeats = 1

    n_patches = patch.__len__()
    dim_intrinsic = intrinsic_process.shape[0]
    dim_sampled = sampled_process.shape[0]

    print_patches(patch, intrinsic_process, color_map, bounding_shape)

    pairs_ind_global = []
    approx_dist_global = []
    real_dist_global = []
    obs_dist_global = []

    # Define Keras DNNs

    sample_patch = Input(shape=(dim_sampled,), name='sample_patch')
    autoencoder_function, low_dim_encoding_function = create_pca_network(dim_sampled=dim_sampled, dim_hid_enc=20, dim_hid_dec=30, dim=2)
    autoencoder_function = autoencoder_function(sample_patch)
    low_dim_encoding_function = low_dim_encoding_function(sample_patch)
    autoencoder_model = Model(input=[sample_patch], output=autoencoder_function)
    low_dim_encoding_model = Model(input=[sample_patch], output=low_dim_encoding_function)
    autoencoder_model.compile(loss='mean_squared_error', optimizer='Adam')
    base_network = create_base_network_sig(dim_sampled=dim_intrinsic, dim_hid=10)
    point_1_input_patch = Input(shape=(dim_intrinsic,), name='point_1_input_patch')
    point_2_input_patch = Input(shape=(dim_intrinsic,), name='point_2_input_patch')
    encoded_a = base_network(point_1_input_patch)
    encoded_b = base_network(point_2_input_patch)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_a, encoded_b])
    rectifier_patch_model = Model(input=[point_1_input_patch, point_2_input_patch], output=distance, name='Distance_Calculator')
    rectifier_patch_model.compile(loss=stress_loss_standard, optimizer='Adam')
    encoder_patch_model = Model(input=[point_1_input_patch], output=encoded_a)

    for i_patch in range(n_patches):

        n_points_in_patch = patch[i_patch].shape[0]
        ind_recon_points_in_patch = numpy.random.choice(patch[i_patch].shape[0], size=n_recon_points, replace=False)
        autoencoder_model.fit([sampled_process[:, patch[i_patch][ind_recon_points_in_patch]].T], [sampled_process[:, patch[i_patch][ind_recon_points_in_patch]].T], nb_epoch=n_epoch_recon, batch_size=n_recon_points, shuffle=True)
        #Check Reconstruction
        patch_reconstruction = autoencoder_model.predict(sampled_process[:, patch[i_patch]].T).T
        print_process(patch_reconstruction[:, ind_recon_points_in_patch], color_map=color_map[patch[i_patch][ind_recon_points_in_patch], :],
                      title='Patch {} Reconstruction'.format(i_patch+1))

        low_dim_patch_encoding_base = low_dim_encoding_model.predict(sampled_process[:, patch[i_patch]].T).T
        low_dim_patch_encoding_step = low_dim_encoding_model.predict(sampled_process[:, patch[i_patch]+1].T).T

        ind_cluster_points_in_patch = numpy.random.choice(n_points_in_patch, size=n_cluster_points_per_patch, replace=False)
        cov_list, cov_neighborhood_list, metric_list_full, metric_list_def = get_metrics([low_dim_patch_encoding_base, low_dim_patch_encoding_step], [sampled_process[:, patch[i_patch]], sampled_process[:, patch[i_patch]+1]], dim_intrinsinc=dim_intrinsinc, ind_cluster_points=ind_cluster_points_in_patch, n_knn = n_knn_cov_est, process_var=process_var)

        pairs_cord_patch, approx_dist_patch = get_jacobian_approximation(low_dim_patch_encoding_base, cov_list, ind_cluster_points_in_patch, dim_intrinsic, process_var)

        pairs_for_patch_rectification_assessment = get_all_index_pairs(n_cluster_points_per_patch)

        n_pairs_for_patch_rectification_assessment = pairs_for_patch_rectification_assessment.shape[0]
        n_used_pairs_for_patch_rectification_assessment = min(n_pairs_for_patch_rectification_assessment, n_max_pairs_for_patch_rectification_assessment)
        ind_pairs_for_patch_rectification_assessment = numpy.random.choice(n_pairs_for_patch_rectification_assessment, size=n_used_pairs_for_patch_rectification_assessment, replace=False)
        pairs_for_patch_rectification_assessment = pairs_for_patch_rectification_assessment[ind_pairs_for_patch_rectification_assessment]

        cords_for_patch_rectification_assessment = get_cord_pairs(pairs_for_patch_rectification_assessment, low_dim_patch_encoding_base[:, ind_cluster_points_in_patch].T)
        obs_dist_before_for_patch_rectification_assessment = get_distance_approx(pairs_for_patch_rectification_assessment, sampled_process[:, patch[i_patch]], ind_cluster_points_in_patch, dist_type='Euclidean', metric_list_full=None, metric_list_def=None)
        #Check Low Dim Representation
        print_process(low_dim_patch_encoding_base[:, ind_cluster_points_in_patch], color_map=color_map[patch[i_patch][ind_cluster_points_in_patch], :], title='Patch {} Low-Dimensional Encoding'.format(i_patch+1), covs=cov_list)
        print_process(low_dim_patch_encoding_base[:, ind_cluster_points_in_patch], color_map=color_map[patch[i_patch][ind_cluster_points_in_patch], :], title='Patch {} Low-Dimensional Encoding Neighborhood Cov'.format(i_patch+1), covs=cov_neighborhood_list)

        approx_dist_for_patch_rectification_assessment = get_distance_approx(pairs_for_patch_rectification_assessment, low_dim_patch_encoding_base, ind_cluster_points_in_patch, dist_type=approx_method, metric_list_full=metric_list_full, metric_list_def=metric_list_def)
        real_dist_for_patch_rectification_assessment = get_distance_approx(pairs_for_patch_rectification_assessment, intrinsic_process[:, patch[i_patch]], ind_cluster_points_in_patch, dist_type='Euclidean')
        rectifier_patch_model.fit([pairs_cord_patch[:, 0, :], pairs_cord_patch[:, 1, :]], approx_dist_patch, nb_epoch=n_epoch_patch_rectifier, validation_data=([cords_for_patch_rectification_assessment[:, 0, :], cords_for_patch_rectification_assessment[:, 1, :]], real_dist_for_patch_rectification_assessment), batch_size=pairs_cord_patch.shape[0], shuffle=True)
        encoded_process_base = encoder_patch_model.predict(low_dim_patch_encoding_base.T).T
        encoded_process_step = encoder_patch_model.predict(low_dim_patch_encoding_step.T).T

        obs_dist_after_for_patch_rectification_assessment = get_distance_approx(pairs_for_patch_rectification_assessment, encoded_process_base, ind_cluster_points_in_patch, dist_type='Euclidean')
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(numpy.asarray(real_dist_for_patch_rectification_assessment), numpy.asarray(real_dist_for_patch_rectification_assessment), c='k')
            ax.scatter(numpy.asarray(real_dist_for_patch_rectification_assessment), numpy.asarray(obs_dist_before_for_patch_rectification_assessment), c='r')
            ax.scatter(numpy.asarray(real_dist_for_patch_rectification_assessment), numpy.asarray(approx_dist_for_patch_rectification_assessment), c='b')
            ax.scatter(numpy.asarray(real_dist_for_patch_rectification_assessment), numpy.asarray(obs_dist_after_for_patch_rectification_assessment), c='g')

            black_patch = mpatches.Patch(color='black', label='Real Intrinsic Distance')
            red_patch = mpatches.Patch(color='red', label='Observed Euclidien Distances Before')
            blue_patch = mpatches.Patch(color='blue', label='Approximated Intrinsic Euclidien Distances')
            green_patch = mpatches.Patch(color='green', label='Observed Euclidien Distances After')
            ax.legend(handles=[black_patch, green_patch, red_patch, blue_patch])
            #plt.axis([0, max(real_dist), 0, 3 * max(real_dist)])
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_ylim(0, ax.get_ylim()[1])
            plt.show(block=False)
        pairs_ind_patch_for_global = get_all_index_pairs(n_cluster_points_per_patch)
        obs_dist_patch_after_rectification = get_distance_approx(pairs_ind_patch_for_global, encoded_process_base, ind_cluster_points_in_patch, dist_type='Euclidean')
        temp_sub_pairs_ind_patch_for_global = arg_crop_by_value(value=obs_dist_patch_after_rectification, percentage=per_dist_approx)
        n_pairs_from_patch_for_global = temp_sub_pairs_ind_patch_for_global.shape[0]
        n_used_pairs_from_patch_for_global = min(n_pairs_from_patch_for_global, n_max_pairs_from_patch_for_global)
        ind_pairs_used_patch_for_global = numpy.random.choice(n_pairs_from_patch_for_global, size=n_used_pairs_from_patch_for_global, replace=False)
        temp_sub_pairs_ind_patch_for_global = temp_sub_pairs_ind_patch_for_global[ind_pairs_used_patch_for_global]
        pairs_ind_patch_for_global = pairs_ind_patch_for_global[temp_sub_pairs_ind_patch_for_global][:][:]
        approx_dist_patch_for_global = obs_dist_patch_after_rectification[temp_sub_pairs_ind_patch_for_global]
        cov_list_after_rectification, cov_neighborhood_list_after_rectification, metric_list_full_after_rectification, metric_list_def_after_rectification = get_metrics([encoded_process_base, encoded_process_step], [sampled_process[:, patch[i_patch]], sampled_process[:, patch[i_patch]+1]], dim_intrinsinc=dim_intrinsinc, ind_cluster_points=ind_cluster_points_in_patch, n_knn = n_knn_cov_est_assessment, process_var=process_var)
        print_process(encoded_process_base[:, ind_cluster_points_in_patch],
                      color_map=color_map[patch[i_patch][ind_cluster_points_in_patch], :],
                      title='Rectified Patch {}'.format(i_patch + 1), covs=cov_list_after_rectification)
        for i_pair in range(pairs_ind_patch_for_global.shape[0]):
            pairs_ind_global += [patch[i_patch][ind_cluster_points_in_patch[pairs_ind_patch_for_global[i_pair][:]]]]
            approx_dist_global += [approx_dist_patch_for_global[i_pair]]

    pairs_ind_global = numpy.asarray(pairs_ind_global)
    pairs_cord_global = get_cord_pairs(pairs_ind_global, sampled_process.T)
    real_dist_global = get_distance_approx(pairs_ind_global, intrinsic_process, dist_type='Euclidean')
    approx_dist_global = numpy.asarray(approx_dist_global)
    real_dist_global = numpy.asarray(real_dist_global)
    base_network_manifold = create_base_network_sig(dim_sampled=dim_sampled, dim_hid=20)
    input_a_manifold = Input(shape=(dim_sampled,), name='point_1_input')
    input_b_manifold = Input(shape=(dim_sampled,), name='point_2_input')
    encoded_a_manifold = base_network_manifold(input_a_manifold)
    encoded_b_manifold = base_network_manifold(input_b_manifold)
    distance_manifold = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_a_manifold, encoded_b_manifold])
    model_manifold = Model(input=[input_a_manifold, input_b_manifold], output=distance_manifold, name='Distance_Calculator')
    model_manifold.compile(loss=stress_loss_standard, optimizer='Adam')
    model_encoder_manifold = Model(input=[input_a_manifold], output=encoded_a_manifold)
    n_total = pairs_cord_global.shape[0]
    n_used_total = min(n_dist_used_manifold_unfolder, n_total)
    ind_dist_used_manifold_unfolder = numpy.random.choice(n_total, size=n_used_total, replace=False)
    pairs_ind_global = pairs_ind_global[ind_dist_used_manifold_unfolder][:]
    approx_dist_global = approx_dist_global[ind_dist_used_manifold_unfolder]
    real_dist_global = real_dist_global[ind_dist_used_manifold_unfolder]

    from mds_smacof import mds_smacof

    for i_repeats in range(0, n_repeats):
        pairs_cord_global = get_cord_pairs(pairs_ind_global, sampled_process.T)
        model_manifold.reset_states()
        model_manifold.fit([pairs_cord_global[:, 0, :], pairs_cord_global[:, 1, :]], approx_dist_global, validation_data=([pairs_cord_global[:, 0, :], pairs_cord_global[:, 1, :]], real_dist_global), nb_epoch=n_epoch_manifold_unfolder, batch_size=numpy.round(n_used_total), shuffle=True)
        encoded_process = model_encoder_manifold.predict(sampled_process.T).T
        obs_dist_global_before = get_distance_approx(pairs_ind_global, sampled_process, dist_type='Euclidean')
        obs_dist_global_after = get_distance_approx(pairs_ind_global, encoded_process, dist_type='Euclidean')
        obs_dist_global_after = numpy.asarray(obs_dist_global_after)
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(numpy.asarray(real_dist_global), numpy.asarray(real_dist_global), c='k')
            ax.scatter(numpy.asarray(real_dist_global), numpy.asarray(obs_dist_global_before), c='r')
            ax.scatter(numpy.asarray(real_dist_global), numpy.asarray(approx_dist_global), c='b')
            ax.scatter(numpy.asarray(real_dist_global), numpy.asarray(obs_dist_global_after), c='g')
            black_patch = mpatches.Patch(color='black', label='Real Intrinsic Distance')
            red_patch = mpatches.Patch(color='red', label='Observed Euclidien Distances Before')
            blue_patch = mpatches.Patch(color='blue', label='Approximated Intrinsic Euclidien Distances')
            green_patch = mpatches.Patch(color='green', label='Observed Euclidien Distances After')
            ax.legend(handles=[black_patch, green_patch, red_patch, blue_patch])
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_ylim(0, ax.get_ylim()[1])
            plt.show(block=False)
        sampled_process = encoded_process


    '''# visualize unfolding
    if dim_sampled == 2 or dim_sampled == 3:
        if dim_sampled == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal', projection='3d')
        line_ani = animation.FuncAnimation(fig, update_lines, frames=nb_epoch, blit=False)
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        line_ani.save('mymovie.mp4', writer=writer)
    '''
    '''# visualize loss graph
    x_iteration = numpy.arange(0, historyLapReg.train.__len__())
    fig, loss_axes = plt.subplots()
    loss_axes.plot(x_iteration, historyLapReg.train, 'b')
    loss_axes.set_xlabel('iteration')
    loss_axes.set_ylabel('loss')
    loss_axes.set_title('Loss vs Iteration')
    loss_axes.set_ylim([0, numpy.max(historyLapReg.train)])
    plt.show(block=False)'''
    #historyLapReg = history

    # plotG(model, to_file='model.png', show_shapes='True')
    # plotG(model_pre, to_file='model_pre.png', show_shapes='True')

    # printing.pydotprint(model, outfile='model_theano.png', var_with_name_simple=True)

    # plotG(model_pre, to_file='model_pre.png')

    #def update_lines(num):
    #    print_process(historyLapReg.encodingEvo[num].T, color_map=color_map, ax=ax, title="Unfolding Process")

    return encoded_process, base_network_manifold

def create_pca_network(dim_sampled, dim_hid_enc=6, dim_hid_dec=10, dim=2):
    activation_type_enc = 'linear'
    activation_type_dec = 'relu'
    # linear, relu, softmax, softplus, softsign, tanh, sigmoid, hard_sigmoid
    input = Input(shape=(dim_sampled,), name='main_input')
    low_dim_encoding = Dense(dim, input_dim=dim_sampled, name='linear_2', activation=activation_type_enc)(input)
    activation_2 = Dense(dim_hid_dec, input_dim=dim, name='linear_3', activation=activation_type_dec)(low_dim_encoding)
    reconstruction = Dense(dim_sampled, input_dim=dim_hid_dec, name='linear_scaling', activation='linear', bias=True)(activation_2)
    return Model(input=[input], output=[reconstruction], name='Autoencoder'), Model(input=[input], output=[low_dim_encoding], name='PCA')


def get_all_index_pairs(n_cluster_points):
    pairs = []
    for i_x in range(0, n_cluster_points):
        for i_y in range(i_x+1, n_cluster_points):
                pairs += [[i_x, i_y]]
    return numpy.asarray(pairs, dtype=numpy.int64)

def get_metrics(low_dim_sampled_process,sampled_process, dim_intrinsinc, ind_cluster_points, n_knn, process_var):

    if sampled_process.__len__() == 2:
        sampled_process_base = sampled_process[0]
        sampled_process_step = sampled_process[1]
        low_dim_sampled_process_base = low_dim_sampled_process[0]
        low_dim_sampled_process_step = low_dim_sampled_process[1]
    else:
        temp = sampled_process[0]
        sampled_process_base = temp[:, 0:-1]
        sampled_process_step = temp[:, 1:]

    n_cluster_points = ind_cluster_points.shape[0]

    # Cluster
    distance_metric = cdist(sampled_process_base[:, ind_cluster_points].T, sampled_process_base.T, 'sqeuclidean')
    #knn_indexes = numpy.argsort(distance_metric, axis=1, kind='quicksort')
    knn_indexes = numpy.argpartition(distance_metric, n_knn)
    #knn_indexes = knn_indexes[:, -n_knn:]
    knn_indexes = knn_indexes[:, 1:n_knn]

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.cla()
    # plt.plot(process[0, :], process[1, :])
    show_ind = 0
    ax.scatter(low_dim_sampled_process_base[0, knn_indexes[show_ind, :]], low_dim_sampled_process_base[1, knn_indexes[show_ind, :]], c='b')
    ax.scatter(low_dim_sampled_process_step[0, knn_indexes[show_ind, :]], low_dim_sampled_process_step[1, knn_indexes[show_ind, :]] ,c='r')

    plt.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show(block=False)'''

    diff_clusters = low_dim_sampled_process_step[:, knn_indexes] - low_dim_sampled_process_base[:, knn_indexes]

    # tangent metric estimation
    metric_list_full = [None] * n_cluster_points
    metric_list_def = [None] * n_cluster_points
    cov_list = [None] * n_cluster_points
    cov_neighborhood_list = [None] * n_cluster_points

    for x in range(0, knn_indexes.shape[0]):
        cov_list[x] = numpy.cov(diff_clusters[:, x, :])
        cov_neighborhood_list[x] = 4*9*numpy.cov(low_dim_sampled_process_base[:, knn_indexes[x, :]])
        U, s, V = numpy.linalg.svd(cov_list[x]/process_var)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsinc:] = float('Inf')
        s_def = 1/s_def
        if s_def[dim_intrinsinc:] < numpy.finfo(numpy.float64).eps:
            s_full[dim_intrinsinc:] = numpy.finfo(numpy.float64).eps
        s_full = 1/s_full
        metric_list_def[x] = numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        metric_list_full[x] = numpy.dot(U, numpy.dot(numpy.diag(s_full), V))

    return cov_list, cov_neighborhood_list, metric_list_full, metric_list_def


def get_distance_approx(pairs, process, ind_cluster_points=None, dist_type='Euclidean', metric_list_full=None, metric_list_def=None):

    if ind_cluster_points is None:
        ind_cluster_points = range(0, process.shape[1])
    # distance estimation and training pair preparation
    n_pairs = pairs.shape[0]
    dist = []
    for i_pair in range(0, n_pairs):
                i_x = pairs[i_pair][0]
                i_y = pairs[i_pair][1]
                if dist_type == 'Euclidean':
                    diff_vect = process[:, ind_cluster_points[i_y]] - process[:, ind_cluster_points[i_x]]
                    dist += [numpy.dot(diff_vect, diff_vect)]
                elif dist_type == 'Mid-Point':
                    temp_vect = numpy.dot(metric_list_full[i_x],
                                          process[:, ind_cluster_points[i_x]]) + numpy.dot(
                        metric_list_full[i_y], process[:, ind_cluster_points[i_y]])
                    mid_point = numpy.dot(numpy.linalg.inv(metric_list_full[i_x] + metric_list_full[i_y]), temp_vect)
                    diff_vect_approx_1 = mid_point - process[:, ind_cluster_points[i_x]]
                    diff_vect_approx_2 = mid_point - process[:, ind_cluster_points[i_y]]
                    temp_approx_dist = numpy.square(numpy.sqrt(
                        numpy.dot(diff_vect_approx_1, numpy.dot(metric_list_def[i_x], diff_vect_approx_1))) + numpy.sqrt(
                        numpy.dot(diff_vect_approx_2, numpy.dot(metric_list_def[i_y], diff_vect_approx_2))))
                    dist += [temp_approx_dist]
                    if False:
                        fig = plt.figure()
                        ax = fig.add_subplot(111, aspect='equal', projection='3d')
                        print_process(process[:, ind_plot_points], color_map=color_map,
                                      title="Observed Process", ax=ax)
                        ax.scatter(process[0, ind_cluster_points[i_x]],
                                   process[1, ind_cluster_points[i_x]],
                                   process[2, ind_cluster_points[i_x]], c='b')
                        ax.scatter(process[0, ind_cluster_points[i_y]],
                                   process[1, ind_cluster_points[i_y]],
                                   process[2, ind_cluster_points[i_y]], c='b')
                        ax.scatter(mid_point[0], mid_point[1], mid_point[2], c='r')
                        ax.plot([process[0, ind_cluster_points[i_x]], mid_point[0]],
                                [process[1, ind_cluster_points[i_x]], mid_point[1]],
                                [process[2, ind_cluster_points[i_x]], mid_point[2]], c='g')
                        ax.plot([process[0, ind_cluster_points[i_y]], mid_point[0]],
                                [process[1, ind_cluster_points[i_y]], mid_point[1]],
                                [process[2, ind_cluster_points[i_y]], mid_point[2]], c='g')

                        plt.axis('equal')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        plt.show(block=False)
                elif dist_type == 'Mahalanobis':
                        diff_vect = process[:, ind_cluster_points[i_y]] - process[:, ind_cluster_points[i_x]]
                        temp_approx_dist = 1 / 2 * (numpy.dot(diff_vect, numpy.dot(metric_list_def[i_x], diff_vect)) + numpy.dot(diff_vect, numpy.dot(metric_list_def[i_y], diff_vect)))
                        dist += [temp_approx_dist]
                else:
                    assert False
    return numpy.asarray(dist)

def create_base_network_sig(dim_sampled, dim_hid=60):
    activation_type = 'sigmoid'
    # linear, relu, softmax, softplus, softsign, tanh, sigmoid, hard_sigmoid
    input = Input(shape=(dim_sampled,), name='main_input')
    linear_1 = Dense(dim_hid, input_dim=dim_sampled, name='linear_1')(input)
    activation_1 = Activation(activation_type, name='activation_1')(linear_1)
    encoding = Dense(dim_sampled, input_dim=dim_hid, name='linear_2')(activation_1)
    #activation_2 = Activation(activation_type, name='activation_2')(linear_2)
    #encoding = Dense(dim_sampled, input_dim=dim_hid, name='linear_3')(activation_2)
    return Model(input=[input], output=[encoding], name='Encoder')


def create_base_network_rect(dim_sampled, dim_hid=60):
    activation_type = 'relu'
    # linear, relu, softmax, softplus, softsign, tanh, sigmoid, hard_sigmoid
    input = Input(shape=(dim_sampled,), name='main_input')
    linear_1 = Dense(dim_hid, input_dim=dim_sampled, name='linear_1')(input)
    activation_1 = Activation(activation_type, name='activation_1')(linear_1)
    encoding = Dense(dim_sampled, input_dim=dim_hid, name='linear_2')(activation_1)
    #activation_2 = Activation(activation_type, name='activation_2')(linear_2)
    #encoding = Dense(dim_sampled, input_dim=dim_hid, name='linear_3')(activation_2)
    return Model(input=[input], output=[encoding], name='Encoder')

class LossHistory(keras.callbacks.Callback):
    def __init__(self, model_pre, sampled_process):
        super().__init__()
        self.train = []
        self.encodingEvo = []
        self.model_pre = model_pre
        self.sampled_process = sampled_process

    def on_train_begin(self, logs={}):
        self.train = []
        self.encodingEvo = []
        #self.train.append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        self.train.append(logs.get('loss'))
        self.encodingEvo.append(self.model_pre.predict(self.sampled_process.T))

def euclidean_distance(inputs):
    u, v = inputs
    return K.sum(K.square(u - v), keepdims=True, axis=1)

def stress_loss_standard(y_true, y_pred):
    #return K.mean(K.abs(K.sqrt(y_true)-K.sqrt(y_pred))/K.sqrt(y_pred), axis=-1)
    return K.mean(K.square(K.sqrt(y_true) - K.sqrt(y_pred)) / y_true, axis=-1)


def stress_loss(y_true, y_pred):
    return K.mean(K.square(K.sqrt(y_true)-K.sqrt(y_pred)), axis=-1)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def arg_crop_by_value(value, percentage):
    sorted_distances = numpy.sort(value, kind='quicksort')
    n_dist = sorted_distances.shape[0]
    sigma = sorted_distances[round(percentage * n_dist) - 1]
    ind_used = numpy.where(value > sigma)
    return ind_used[0]

def get_cord_pairs(pairs_ind, points):
    pairs_cord = []
    points = points.T
    for i_pair in range(pairs_ind.shape[0]):
        pairs_cord += [[points[:, pairs_ind[i_pair, 0]], points[:, pairs_ind[i_pair, 1]]]]
    pairs_cord = numpy.asarray(pairs_cord)
    return pairs_cord

def print_patches(patches, intrinsic_process, color_map, bounding_shape, n_patch_plot_points = 2000, n_patches_to_print = 9):
    n_patches = patches.__len__()
    dim_intrinsic = intrinsic_process.shape[0]
    if dim_intrinsic == 2 or dim_intrinsic == 3:
        n_patches_to_print = min(n_patches, 16)
        n_rows_cols = numpy.int8(numpy.ceil(numpy.sqrt(n_patches_to_print)))

    if dim_intrinsic == 2:
        x_min = numpy.min(intrinsic_process[0, :])
        y_min = numpy.min(intrinsic_process[1, :])
        x_max = numpy.max(intrinsic_process[0, :])
        y_max = numpy.max(intrinsic_process[1, :])
        fig = plt.figure()

    for i_patch in range(n_patches):

        n_points_in_patch = patches[i_patch].shape[0]

        if i_patch <= n_patches_to_print:
            ax = fig.add_subplot(n_rows_cols, n_rows_cols, i_patch + 1)
            print_ind_in_patch = numpy.random.choice(n_points_in_patch, size=n_patch_plot_points, replace=False)
            print_ind = patches[i_patch][print_ind_in_patch]
            print_process(intrinsic_process[:, print_ind], color_map=color_map[print_ind, :],
                          title='Patch Num {}'.format(i_patch + 1), ax=ax, bounding_shape=bounding_shape)

            if dim_intrinsic == 2:
                ax.set_xlim((x_min, x_max))
                ax.set_ylim((y_min, y_max))

def get_patches(observed_process, n_patches):
    n_patches_keep = n_patches
    per_points_in_cluster = min(1, 1)
    n_trajectory_points = observed_process.shape[1]
    # Select cluster centers
    n_cluster_points = 2000
    ind_cluster_center_points = numpy.random.choice(n_trajectory_points, size=n_cluster_points, replace=False)
    distance_metric = cdist(observed_process[:, ind_cluster_center_points].T,
                            observed_process[:, ind_cluster_center_points].T, 'sqeuclidean')
    sigma = 0.01
    affinity = numpy.exp(-distance_metric / (2 * sigma ** 2))
    sum_row = numpy.sum(affinity, axis=1)
    sum_row = 1 / sum_row
    diff_mat = numpy.dot(numpy.diag(sum_row), affinity)
    sum_diff_mat_check = numpy.sum(diff_mat, axis=1)
    diff_time = 10
    diff_mat_prop = numpy.linalg.matrix_power(diff_mat, diff_time)
    sum_diff_mat_check = numpy.sum(diff_mat_prop, axis=1)

    n_patches_try = 100
    ind_patch_center_points = numpy.random.choice(n_cluster_points, size=n_patches_try, replace=False)
    n_clusters_in_patch = numpy.int32(numpy.ceil(per_points_in_cluster * n_cluster_points))
    in_patches = numpy.argsort(diff_mat_prop[ind_patch_center_points, :], axis=1)
    in_patches = in_patches[:, -n_clusters_in_patch:]
    X = []
    y = []

    for i_cluster in range(n_cluster_points):
        X += [observed_process[:, ind_cluster_center_points[i_cluster]].T]
        y += [i_cluster]

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X, y)
    cluster_assosiation = neigh.predict(observed_process[:, :-1].T)

    # knn_assosiation = knn_classification(observed_process, observed_process[:, ind_cluster_center_points], ind_cluster_center_points)
    # knn_assosiation_patchs = knn_assosiation

    patch = []
    for i_patch in range(n_patches_try):
        points_in_patch = []
        for i_cluster in range(n_clusters_in_patch):
            cluster_num = in_patches[i_patch, i_cluster]
            points_in_patch += numpy.where(cluster_assosiation == cluster_num)
        points_in_patch = numpy.concatenate(points_in_patch)
        patch += [points_in_patch]

    best_so_far = 0
    n_trys = 1000
    for i_try in range(0, n_trys):
        ind_patches = numpy.random.choice(n_patches_try, size=n_patches_keep, replace=False)
        flag_used = numpy.zeros(n_trajectory_points - 1)
        for i_keep in range(0, n_patches_keep):
            flag_used[patch[ind_patches[i_keep]]] = 1
        score = numpy.sum(flag_used)
        if score > best_so_far:
            best_so_far = score
            ind_patches_best = ind_patches

    patch_final = []
    for i_patch in range(n_patches_keep):
        patch_final += [patch[ind_patches_best[i_patch]]]
    return patch_final

def get_jacobian_approximation(process, cov_list, ind_cluster_points_in_patch, dim_intrinsic, process_var):
    process_std = numpy.sqrt(process_var)
    step = 1e-1
    cord_pairs = []
    approx_dist = []
    n_clusters = ind_cluster_points_in_patch.shape[0]

    for i_cluster in range(0, n_clusters):
        i_cov = cov_list[i_cluster]
        U, s, V = numpy.linalg.svd(i_cov)
        base_point = process[:, ind_cluster_points_in_patch[i_cluster]]
        for i_dim in range(0, dim_intrinsic):
            cord_pairs += [[base_point, base_point + step*numpy.sqrt(s[i_dim])*V[:, i_dim]]]
            approx_dist += [(process_std*step)**2]
    cord_pairs = numpy.asarray(cord_pairs)
    return cord_pairs, numpy.asarray(approx_dist)
