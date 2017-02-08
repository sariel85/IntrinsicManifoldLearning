import scipy
from scipy.spatial.distance import cdist
import numpy
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import manifold
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets



def get_short_range_dist(sampled_process, ind_cluster_center_points, n_knn, intrinsic_process, approximated, k=5, process_var=1):
    n_cluster_points = ind_cluster_center_points.__len__()
    dim_sampled = sampled_process.shape[0]
    dim_intrinsic = intrinsic_process.shape[0]

    # cluster
    distance_metric = cdist(sampled_process[:, ind_cluster_center_points].T, sampled_process[:, :-1].T, 'sqeuclidean')
    knn_indexes = numpy.argsort(distance_metric, axis=1, kind='quicksort')
    knn_indexes = knn_indexes[:, 1:n_knn]

    diff_clusters = sampled_process[:, knn_indexes + 1] - sampled_process[:, knn_indexes]

    # tangent metric estimation
    cov_list_full = [None] * n_cluster_points
    cov_list_def = [None] * n_cluster_points

    for x in range(0, knn_indexes.shape[0]):
        temp_cov = numpy.cov(diff_clusters[:, x, :])
        U, s, V = numpy.linalg.svd(temp_cov)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        s_def = 1/s_def
        if s_def[dim_intrinsic:] < numpy.finfo(numpy.float32).eps:
            s_full[dim_intrinsic:] = numpy.finfo(numpy.float32).eps

        s_full = 1/s_full
        cov_list_def[x] = numpy.dot(U,numpy.dot(numpy.diag(s_def), V))
        cov_list_full[x] = numpy.dot(U,numpy.dot(numpy.diag(s_full), V))

    # distance estimation and training pair preparation

    approx_dist = numpy.zeros((n_cluster_points, n_cluster_points))

    distFull = numpy.zeros((n_cluster_points, n_cluster_points))
    distDef = numpy.zeros((n_cluster_points, n_cluster_points))

    for i_x in range(0, knn_indexes.shape[0]):
        for i_y in range(0, knn_indexes.shape[0]):
            if i_x != i_y:

                distFull[i_x, i_y] = 0
                distDef[i_x, i_y] = 0

                if approximated:

                    #temp_vect = numpy.dot(cov_list_full[i_x], sampled_process[:, ind_cluster_center_points[i_x]]) + numpy.dot(cov_list_full[i_y], sampled_process[:, ind_cluster_center_points[i_y]])
                    #mid_point = numpy.dot(numpy.linalg.inv(cov_list_full[i_x] + cov_list_full[i_y]), temp_vect)
                    #dif_vect1 = mid_point - sampled_process[:, ind_cluster_center_points[i_x]]
                    #dif_vect2 = mid_point - sampled_process[:, ind_cluster_center_points[i_y]]

                    #curr_dist = numpy.square(numpy.sqrt(numpy.dot(dif_vect1, numpy.dot(cov_list_full[i_x], dif_vect1))) + numpy.sqrt(numpy.dot(dif_vect2, numpy.dot(cov_list_full[i_y], dif_vect2))))
                    #curr_dist = numpy.dot(dif_vect1, numpy.dot(cov_list_def[i_x], dif_vect1)) + numpy.dot(dif_vect2, numpy.dot(cov_list_def[i_y], dif_vect2))

                    dif_vect = sampled_process[:, ind_cluster_center_points[i_x]] - sampled_process[:, ind_cluster_center_points[i_y]]

                    distFull[i_x, i_y] = process_var*1/2*(numpy.dot(dif_vect, numpy.dot(cov_list_full[i_x], dif_vect)) + numpy.dot(dif_vect, numpy.dot(cov_list_full[i_y], dif_vect)))
                    distDef[i_x, i_y] = process_var*1/2*(numpy.dot(dif_vect, numpy.dot(cov_list_def[i_x], dif_vect)) + numpy.dot(dif_vect, numpy.dot(cov_list_def[i_y], dif_vect)))


                    '''dif_vect = intrinsic_process[:, ind_cluster_center_points[i_x]] - intrinsic_process[:, ind_cluster_center_points[i_y]]
                    curr_dist = numpy.dot(dif_vect, dif_vect)
                    labels += [curr_dist]
                    weights += [curr_dist]'''
                else:
                    dif_vect = intrinsic_process[:, ind_cluster_center_points[i_x]] - intrinsic_process[:, ind_cluster_center_points[i_y]]
                    curr_dist = numpy.dot(dif_vect, dif_vect)
                    distFull[i_x, i_y] = curr_dist
                    distDef[i_x, i_y] = curr_dist

    n_dist = distFull.shape[1]

    for i_x in range(0, knn_indexes.shape[0]):
        sortedDistances = numpy.sort(distFull[i_x, :], kind='quicksort')
        sigma = sortedDistances[k]
        ind_used = numpy.where(distFull[i_x, :] > sigma)
        distDef[i_x, ind_used] = 0


    return numpy.sqrt(distDef)


def print_potential(func, x_low=-0.25, x_high=0.25, y_low=-0.25, y_high=0.25, step=0.01):
    x_range = numpy.arange(x_low, x_high, step)
    n_x = x_range.shape[0]
    y_range = numpy.arange(y_low, y_high, step)
    n_y = y_range.shape[0]
    [X_grid, Y_grid] = numpy.meshgrid(x_range, y_range)
    potential = numpy.empty((n_x, n_y))
    for i in range(0, n_x):
        for j in range(0, n_y):
            potential[i, j] = func(numpy.asarray([X_grid[j, i], Y_grid[j, i]]))

    plt.figure()

    plt.contour(X_grid.T, Y_grid.T, potential, 15, linewidths=0.5, colors='k')
    plt.contourf(X_grid.T, Y_grid.T, potential, 15, cmap=plt.cm.rainbow, vmin=potential.min(), max=potential.max())
    plt.colorbar()  # draw colorbar
    ax = plt.gca()
    ax.set_title("Potential Level Contours")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X_grid.T, Y_grid.T, potential, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_title("Potential Surface")

    plt.show(block=False)


def embbeding_score(ground_truth, embedding, titleStr, n_points = 200):
    ground_truth = ground_truth[:, :n_points]
    embedding = embedding[:, :n_points]
    dist_mat_ground_truth = scipy.spatial.distance.cdist(ground_truth.T, ground_truth.T)
    dist_mat_embedding = scipy.spatial.distance.cdist(embedding.T, embedding.T)
    stress = numpy.sqrt(numpy.sum(numpy.square(dist_mat_ground_truth-dist_mat_embedding)))
    stress_norm = numpy.sqrt(numpy.sum(numpy.square(dist_mat_ground_truth)))
    stress_normalized = stress/stress_norm
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(dist_mat_ground_truth.reshape((1, n_points*n_points)), dist_mat_embedding[:].reshape((1, n_points*n_points)), s=1)
    ax.plot([0, numpy.max(dist_mat_ground_truth)], [0, numpy.max(dist_mat_ground_truth)], c='r')
    ax.set_xlim([0, numpy.max(dist_mat_ground_truth)*1.15])
    ax.set_ylim([0, numpy.max(dist_mat_ground_truth)*1.15])
    ax.set_xlabel('Ground Truth Distances')
    ax.set_ylabel('Distance in Recovered Embedding')
    ax.set_title(titleStr)
    #plt.axis('equal')
    plt.show(block=False)

    return stress, stress_normalized

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform(A_in, B_in):
    assert len(A_in) == len(B_in)

    if A_in.shape[0] == 2:
        A = numpy.pad(A_in, pad_width=((0, 1), (0, 1)), mode='constant')
        B = numpy.pad(B_in, pad_width=((0, 1), (0, 1)), mode='constant')

    N = A.shape[1]  # total points

    A = A.T
    B = B.T

    centroid_A = numpy.mean(A, axis=0)

    centroid_B = numpy.mean(B, axis=0)

    # centre the points
    AA = A - numpy.tile(centroid_A, (N, 1))
    BB = B - numpy.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = numpy.dot(numpy.transpose(AA), BB)

    U, S, Vt = numpy.linalg.svd(H)

    R = numpy.dot(Vt.T, U.T)

    # special reflection case
    if numpy.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = numpy.dot(Vt.T, U.T)
    t = -numpy.dot(R, centroid_A.T) + centroid_B.T


    if A_in.shape[0] == 2:
        R = R[0:2, 0:2]
        t = t[0:2]

    return R, t

def calc_dist(points, metrics=None):
    dim_points = points.shape[0]
    n_points = points.shape[1]
    dist_mat = numpy.zeros((n_points, n_points))

    if metrics is None:
        metrics = [None] * n_points
        for i_x in range(0, n_points):
            metrics[i_x] = numpy.eye(dim_points)

    for i_x in range(0, n_points):
            tmp1 = numpy.dot(metrics[i_x], points[:, i_x])
            a2 = numpy.dot(points[:, i_x].T, tmp1)
            b2 = sum(points * numpy.dot(metrics[i_x], points), 0)
            ab = numpy.dot(points.T, tmp1)
            dist_mat[:, i_x] = numpy.real((numpy.tile(a2, (n_points, 1)) + b2.T.reshape((n_points, 1)) - 2 * ab.reshape((n_points, 1))).reshape(n_points))

    dist_mat = numpy.abs((dist_mat + dist_mat.T)/2)

    return dist_mat

def get_metrics_from_points(cluster_centers, input_base, input_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance, measurement_variance):

    n_cluster_points = cluster_centers.shape[1]
    distance_metrix = scipy.spatial.distance.cdist(cluster_centers.T, input_base.T, metric='sqeuclidean')
    knn_indexes = numpy.argsort(distance_metrix, axis=1, kind='quicksort')
    knn_indexes = knn_indexes[:, 1:n_neighbors_cov+1]

    diff_clusters = input_step[:, knn_indexes] - input_base[:, knn_indexes]

    cov_list_def = [None] * n_cluster_points
    cov_list_full = [None] * n_cluster_points

    for x in range(0, knn_indexes.shape[0]):
        temp_cov = numpy.cov(diff_clusters[:, x, :])
        U, s, V = numpy.linalg.svd(temp_cov)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        s_def = 1 / s_def
        if s_def[dim_intrinsic:] < numpy.finfo(numpy.float64).eps:
            s_full[dim_intrinsic:] = numpy.max(numpy.finfo(numpy.float64).eps, measurement_variance)

        s_full = 1 / s_full
        cov_list_def[x] = intrinsic_variance*numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        cov_list_full[x] = intrinsic_variance*numpy.dot(U, numpy.dot(numpy.diag(s_full), V))


    return cov_list_def, cov_list_full

def get_metrics_from_points_static(noisy_sensor_measured, dim_intrinsic, intrinsic_variance, measurement_variance):

    dim_measured = noisy_sensor_measured.shape[0]
    n_cluster_points = int(noisy_sensor_measured.shape[1]/(dim_intrinsic+1))

    cov_list_def = [None] * n_cluster_points
    cov_list_full = [None] * n_cluster_points

    noisy_sensor_measured = noisy_sensor_measured.T
    for i_cluster in range(n_cluster_points):
        J_temp = numpy.zeros((dim_measured, dim_intrinsic))
        for i_dim in range(dim_intrinsic):
            J_temp[:, i_dim] = (noisy_sensor_measured[i_cluster*(dim_intrinsic+1) + i_dim + 1, :] - noisy_sensor_measured[i_cluster*(dim_intrinsic+1), :])/numpy.sqrt(intrinsic_variance)

        temp_cov = numpy.dot(J_temp, J_temp.T)
        U, s, V = numpy.linalg.svd(temp_cov)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        s_def = 1 / s_def

        if s_def[dim_intrinsic:] < numpy.finfo(numpy.float64).eps:
            s_full[dim_intrinsic:] = numpy.max(numpy.finfo(numpy.float64).eps, measurement_variance)

        s_full = 1 / s_full
        cov_list_def[i_cluster] = numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        cov_list_full[i_cluster] = numpy.dot(U, numpy.dot(numpy.diag(s_full), V))
    noisy_sensor_measured = noisy_sensor_measured.T
    return cov_list_def, cov_list_full



def get_metrics_from_net(non_local_tangent_net, noisy_sensor_clusters):
    dim_measurement = noisy_sensor_clusters.shape[0]
    n_points = noisy_sensor_clusters.shape[1]

    metric_list_net_tangent = [None] * n_points
    metric_list_net_intrinsic = [None] * n_points

    for i_point in range(0, n_points):
        jacobian = non_local_tangent_net.get_jacobian_val(noisy_sensor_clusters[:, i_point].reshape((dim_measurement, 1)))[0, :, :]
        jacobian_int = non_local_tangent_net.get_jacobian_int_val(noisy_sensor_clusters[:, i_point].reshape((dim_measurement, 1)))[0, :, :]
        jacobian_total = numpy.dot(jacobian, numpy.linalg.pinv(jacobian_int))
        metric_list_net_tangent[i_point] = numpy.linalg.pinv(numpy.dot(jacobian, jacobian.T))
        metric_list_net_intrinsic[i_point] = numpy.linalg.pinv(numpy.dot(jacobian_total, jacobian_total.T))

    return metric_list_net_tangent, metric_list_net_intrinsic

def get_drift_from_net(non_local_tangent_net, noisy_sensor_clusters):
        drift = non_local_tangent_net.get_drift_val(noisy_sensor_clusters)
        return drift

def trim_distances(dist_mat, dist_mat_criteria=None, n_neighbors=10):

    n_points = dist_mat.shape[0]

    if dist_mat_criteria is None:
        dist_mat_criteria = dist_mat
    if n_neighbors is None:
        n_neighbors = numpy.ceil(n_points*0.1)

    n_points = dist_mat.shape[0]
    knn_indexes = numpy.argsort(dist_mat_criteria, axis=1, kind='quicksort')
    knn_indexes = knn_indexes[:, 1:n_neighbors + 1]
    dist_mat_trim = numpy.zeros((n_points, n_points))
    for i_x in range(0, n_points):
        for i_y in range(0, n_neighbors):
            dist_mat_trim[i_x, knn_indexes[i_x, i_y]] = dist_mat[i_x, knn_indexes[i_x, i_y]]
            dist_mat_trim[knn_indexes[i_x, i_y], i_x] = dist_mat[i_x, knn_indexes[i_x, i_y]]
    return dist_mat_trim

def trim_distances_topo(dist_mat, dist_potential, radius_trim, intrinsic_process):

    n_points = dist_mat.shape[0]
    dist_mat_trim = numpy.zeros(dist_mat.shape)

    [dist_mat, predecessors]  = scipy.sparse.csgraph.shortest_path(dist_mat, directed=False, return_predecessors=True, method='D')

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(intrinsic_process[0, numpy.where(dist_potential<radius_trim)], intrinsic_process[1, numpy.where(dist_potential<radius_trim)], c="r")

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c="k")

    for i_x in range(0, n_points):
        for i_y in range(0, n_points):
            pre_temp = i_y
            edge_list = []
            while pre_temp != -9999:
                edge_list.append(dist_potential[pre_temp]<radius_trim)
                pre_temp = predecessors[i_x, pre_temp]
            if all(edge == False for edge in edge_list):
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]
            elif (edge_list[0]== True) and all(edge == False for edge in edge_list[1:]):
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]
            elif (edge_list[-1]== True) and all(edge == False for edge in edge_list[:-1]):
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]
            elif (edge_list[1]== True) and (edge_list[-1]== True) and all(edge == False for edge in edge_list[1:-1]):
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c="k")

    for i_x in range(0, n_points):
        for i_y in range(0, n_points):
            if (dist_mat_trim[i_x, i_y]!=0):
                ax.plot(intrinsic_process[0, [i_x,i_y]], intrinsic_process[1, [i_x,i_y]], '-', c='k')


    #for i_x in range(0, n_points):
    #    for i_y in range(0, n_points):
    #        if (dist_mat_trim[i_x, i_y]==0):
    #            ax.plot(intrinsic_process[0, [i_x,i_y]], intrinsic_process[1, [i_x,i_y]], '-', c='k')

    return dist_mat_trim



def calc_diff_map(dist_mat, dims=2, factor=2):
    sigma = numpy.median(dist_mat)/factor
    diff_kernal = numpy.exp(-(dist_mat ** 2) / (2 * sigma ** 2))
    row_sum = numpy.sum(diff_kernal, axis=1)
    normlized_kernal = numpy.dot(numpy.diag(1 / row_sum), diff_kernal)
    U, S, V = numpy.linalg.svd(normlized_kernal)
    return U[:, 1:dims+1]

def print_metrics(noisy_sensor_clusters, metric_list_full, intrinsic_dim, titleStr, scale, space_mode, elipse, color_map):
    metric_list = []
    n_metrics_to_print = 200

    n_points = noisy_sensor_clusters.shape[1]

    n_points_used_for_clusters = min(n_points, n_metrics_to_print)
    points_used_for_clusters_indexs = numpy.random.choice(n_points, size=n_points_used_for_clusters, replace=False)

    noisy_sensor_clusters = numpy.copy(noisy_sensor_clusters[:, points_used_for_clusters_indexs])
    for i_point in range(n_points_used_for_clusters):
        metric_list.append(metric_list_full[points_used_for_clusters_indexs[i_point]])

    color_map = color_map[points_used_for_clusters_indexs, :]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    n_points = noisy_sensor_clusters.shape[1]


    if space_mode is True:
        for i_point in range(0, n_points):
            metric = metric_list[i_point]
            center = noisy_sensor_clusters[:, i_point]

            if elipse:
                U, s, rotation = numpy.linalg.svd(metric)
                # now carry on with EOL's answer
                u = numpy.linspace(0.0, 2.0 * numpy.pi, 12)
                v = numpy.linspace(0.0, numpy.pi, 6)
                x = int(s[0]>1e-10)*numpy.outer(numpy.cos(u), numpy.sin(v))
                y = int(s[1]>1e-10)*numpy.outer(numpy.sin(u), numpy.sin(v))
                z = (int(s[2]>1e-10)+1e-5)*numpy.outer(numpy.ones_like(u), numpy.cos(v))
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        [x[i, j], y[i, j], z[i, j]] = numpy.dot([x[i, j], y[i, j], z[i, j]], numpy.sqrt(scale)*rotation) + center

                ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color=color_map[i_point], alpha=1)
            else:
                [u, s, v] = numpy.linalg.svd(metric)
                u = numpy.dot(u[:, 0:intrinsic_dim], numpy.diag(numpy.sqrt(1/s[:intrinsic_dim])))
                sign = numpy.sign(u[0, 0])
                ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                          noisy_sensor_clusters[2, i_point], sign*u[0, 0], sign*u[1, 0], sign*u[2, 0],
                          length=3*numpy.sqrt(scale), pivot='tail')
                sign = numpy.sign(u[0, 1])
                ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                          noisy_sensor_clusters[2, i_point], sign*u[0, 1], sign*u[1, 1], sign*u[2, 1],
                          length=3*numpy.sqrt(scale), pivot='tail')

    else:

        for i_point in range(0, n_points):
            metric = metric_list[i_point]
            center = noisy_sensor_clusters[:, i_point]

            if elipse:
                U, s, rotation = numpy.linalg.svd(metric)
                radii = (1.0 / numpy.sqrt(s))/3
                radii[intrinsic_dim:] = 0
                # now carry on with EOL's answer
                u = numpy.linspace(0.0, 2.0 * numpy.pi, 12)
                v = numpy.linspace(0.0, numpy.pi, 6)
                x = radii[0] * numpy.outer(numpy.cos(u), numpy.sin(v))
                y = radii[1] * numpy.outer(numpy.sin(u), numpy.sin(v))
                z = radii[2] * numpy.outer(numpy.ones_like(u), numpy.cos(v))
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        [x[i, j], y[i, j], z[i, j]] = numpy.dot([x[i, j], y[i, j], z[i, j]], numpy.sqrt(scale)*rotation) + center

                ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color=color_map[i_point], alpha=1)
            else:
                [u, s, v] = numpy.linalg.svd(metric)
                u = numpy.dot(u[:, 0:intrinsic_dim], numpy.sqrt(scale)*numpy.diag(numpy.sqrt(1/s[:intrinsic_dim])))
                sign = numpy.sign(u[0, 0])
                ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                          noisy_sensor_clusters[2, i_point], sign*u[0, 0], sign*u[1, 0], sign*u[2, 0],
                          length=numpy.linalg.norm(u[:, 0]), pivot='tail')
                sign = numpy.sign(u[0, 1])
                ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                          noisy_sensor_clusters[2, i_point], sign*u[0, 1], sign*u[1, 1], sign*u[2, 1],
                          length=numpy.linalg.norm(u[:, 1]), pivot='tail')
    ax.set_title(titleStr)

def print_drift(noisy_sensor_clusters, drift, titleStr):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    n_points = noisy_sensor_clusters.shape[1]
    for i_point in range(0, n_points):
        ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                  noisy_sensor_clusters[2, i_point], drift[i_point, 0], drift[i_point, 1], drift[i_point, 2], length=numpy.linalg.norm(drift[i_point, :]), pivot='tail')
    ax.set_title(titleStr)

def trim_non_euc(dist_mat_fill, dist_mat_trust, dim_intrinsic, intrinsic_process_clusters):

    n_points = dist_mat_trust.shape[0]
    dist_mat_trimmed = numpy.zeros((n_points, n_points))
    dist_mat_trimmed_wgt = numpy.zeros((n_points, n_points))
    #indexs_balls = numpy.random.choice(n_points, size=n_points, replace=False)

    for i_point in range(15):
        dist_mat_trust_temp = numpy.array(dist_mat_trust, copy=True)

        knn_indexes = numpy.argsort(dist_mat_fill[i_point], kind='quicksort')
        D_sub_trust_original = dist_mat_fill[knn_indexes, :][:, knn_indexes]

        plt.figure()
        plt.imshow(D_sub_trust_original, vmin=numpy.min(D_sub_trust_original), vmax=numpy.max(D_sub_trust_original))

        n_neighbors_start = 30
        n_neighbors_step = 30

        n_neighbors = n_neighbors_start

        flat = True
        check_list = []
        check_list_X = []
        while flat:
            knn_indexes_sub = knn_indexes[0:n_neighbors]

            numpy.fill_diagonal(dist_mat_fill, 0)

            D_fill_sub = dist_mat_fill[knn_indexes_sub, :][:, knn_indexes_sub]

            #plt.figure()
            #plt.imshow(D_fill_sub, vmin=numpy.min(D_sub_trust_original), vmax=numpy.max(D_sub_trust_original))

            # square it
            D_squared = D_fill_sub ** 2

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

            #eigen_vect = eigen_vect[eigen_val_sort_ind]
            #eigen_vect = eigen_vect[:dim_intrinsic].T
            #guess = numpy.real(numpy.dot(numpy.diag(numpy.sqrt(numpy.abs(eigen_val[:dim_intrinsic]))), eigen_vect.T).T)

            #embedding_dist = numpy.sqrt(calc_dist(guess.T))

            #plt.figure()
            #plt.imshow(embedding_dist, vmin=numpy.min(D_sub_trust_original), vmax=numpy.max(D_sub_trust_original))

            #embedding_dist_diff = embedding_dist - D_fill_sub
            #plt.figure()
            #plt.imshow(embedding_dist_diff, vmin=numpy.min(embedding_dist_diff), vmax=numpy.max(embedding_dist_diff))

            #plt.figure()
            #plt.stem(eigen_val[:10])
            #plt.show(block=False)


            #guess = numpy.real(numpy.dot(numpy.diag(numpy.sqrt(numpy.abs(eigen_val[:dim_intrinsic]))), eigen_vect.T).T)

            #wgt = (D_sub_trust_original != 0).astype(int)

            #mds = manifold.MDS(n_components=dim_intrinsic, max_iter=2000, eps=1e-7, dissimilarity="precomputed", n_jobs=1, n_init=1)
            #flat_local = mds.fit(D_fill_sub, init=guess[:, 0:dim_intrinsic,]).embedding_
            #stress1 = mds.stress_/numpy.sum(D_squared)
            #check_list.append(stress1)

            #flat_local = mds.fit(D_sub_trust_original, weight=wgt, init=guess).embedding_
            #stress2 = mds.stress_

            #flat_local = flat_local.T
            #fig = plt.figure()
            #ax = fig.gca()
            #ax.scatter(flat_local[0, :], flat_local[1, :], c="k")

            expl = numpy.sum(eigen_val[:dim_intrinsic])
            res = numpy.sum(eigen_val[dim_intrinsic:])
            #dis = (D_sub_trust_original*wgt).sum()
            #check = (stress2/dis)
            check_list.append(dim_intrinsic*eigen_val[dim_intrinsic]/expl)
            #check_list.append((eigen_val[0]-eigen_val[dim_intrinsic-1]+eigen_val[dim_intrinsic])/(eigen_val[dim_intrinsic-1]-eigen_val[dim_intrinsic]))
            #check_list.append(numpy.mean(eigen_val[:dim_intrinsic])/eigen_val[dim_intrinsic]-eigen_val[dim_intrinsic]/eigen_val[dim_intrinsic+1])

            check_list_X.append(n_neighbors)

            #flat = (check < 0.05)

            '''
            guess = guess.T
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(guess[0, 0], guess[1, 0], guess[2, 0], c="g")
            ax.scatter(guess[0, 1:], guess[1, 1:], guess[2, 1:], c="r")
            plt.axis('equal')
            plt.show(block=False)
            plt.title("Classical MDS")
            plt.axis('equal')
            '''

            #flat_local = flat_local.T

            #fig = plt.figure()
            #ax = fig.gca()
            #ax.scatter(flat_local[0, 0], flat_local[1, 0], c="g")
            #ax.scatter(flat_local[0, 1:], flat_local[1, 1:], c="r")
            #plt.axis('equal')
            #plt.show(block=False)
            #plt.title("After LS-MDS Correction")
            #plt.axis('equal')

            #plt.figure()
            #plt.plot(numpy.asarray(check_list_X), numpy.asarray(check_list))
            #plt.show(block=False)

            if n_neighbors == n_points:
                break
            #dis = numpy.sqrt(calc_dist(flat_local))
            #for i_row in range(knn_indexes_sub.shape[0]):
            #    for i_col in range(knn_indexes_sub.shape[0]):
            #        dist_mat_trust_temp[knn_indexes_sub[i_row], knn_indexes_sub[i_col]] = dis[i_row, i_col]

            #plt.figure()
            #plt.plot(numpy.asarray(check_list_X), numpy.asarray(check_list))
            #plt.show(block=False)

            n_neighbors = min(numpy.ceil(n_neighbors+n_neighbors_step), n_points)

        '''
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(intrinsic_process_clusters[0, :], intrinsic_process_clusters[1, :], c="k")
        for j_point in knn_indexes_sub:
            ax.scatter(intrinsic_process_clusters[0, j_point], intrinsic_process_clusters[1, j_point], c='r')
        ax.scatter(intrinsic_process_clusters[0, knn_indexes_sub[0]], intrinsic_process_clusters[1, knn_indexes_sub[0]], c='g')

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(flat_local[0, 0], flat_local[1, 0], c="g")
        ax.scatter(flat_local[0, 1:], flat_local[1, 1:], c="r")
        plt.axis('equal')
        plt.show(block=False)
        '''

        plt.figure()
        plt.plot(numpy.asarray(check_list_X), numpy.asarray(check_list))
        plt.show(block=False)

        min_rank = numpy.asarray(check_list).min()

        if (min_rank>0.05):
            break

        min_rank_flex = min_rank*1.2

        ind_neigboors = numpy.where(check_list<min_rank_flex)[-1]

        radius = check_list_X[ind_neigboors[-1]]

        print(i_point)

        knn_indexes_sub = knn_indexes[0:radius]

        numpy.fill_diagonal(dist_mat_fill, 0)

        D_fill_sub = dist_mat_fill[knn_indexes_sub, :][:, knn_indexes_sub]

        # plt.figure()
        # plt.imshow(D_fill_sub, vmin=numpy.min(D_sub_trust_original), vmax=numpy.max(D_sub_trust_original))

        # square it
        D_squared = D_fill_sub ** 2

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
        guess = numpy.real(numpy.dot(numpy.diag(numpy.sqrt(numpy.abs(eigen_val[:dim_intrinsic]))), eigen_vect.T).T)

        # wgt = (D_sub_trust_original != 0).astype(int)

        mds = manifold.MDS(n_components=dim_intrinsic, max_iter=2000, eps=1e-7, dissimilarity="precomputed", n_jobs=1, n_init=1)
        flat_local = mds.fit(D_fill_sub, init=guess[:, 0:dim_intrinsic,]).embedding_

        flat_local = flat_local.T

        embedding_dist = numpy.sqrt(calc_dist(flat_local))

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(intrinsic_process_clusters[0, :], intrinsic_process_clusters[1, :], c="k")
        for j_point in knn_indexes_sub:
            ax.scatter(intrinsic_process_clusters[0, j_point], intrinsic_process_clusters[1, j_point], c='r')
        ax.scatter(intrinsic_process_clusters[0, knn_indexes_sub[0]], intrinsic_process_clusters[1, knn_indexes_sub[0]], c='g')
        plt.axis('equal')

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(flat_local[0, :], flat_local[1, :], c="k")
        for j_point in range(flat_local.shape[1]):
            ax.scatter(flat_local[0, j_point], flat_local[1, j_point], c='r')
        ax.scatter(flat_local[0, knn_indexes_sub[0]], flat_local[1, knn_indexes_sub[0]], c='g')
        plt.axis('equal')
        #dist_mat_trimmed = dist_mat_trimmed + dist_mat_trust_temp

        #dist_mat_trimmed_wgt = dist_mat_trimmed_wgt +1
        for i_row in range(knn_indexes_sub.shape[0]):
            for i_col in range(knn_indexes_sub.shape[0]):
                dist_mat_trimmed[knn_indexes_sub[i_row], knn_indexes_sub[i_col]] = dist_mat_trimmed[i_row, i_col] + embedding_dist[i_row, i_col]
                dist_mat_trimmed_wgt[knn_indexes_sub[i_row], knn_indexes_sub[i_col]] = dist_mat_trimmed_wgt[i_row, i_col] + 1


    dist_mat_trimmed = dist_mat_trimmed/numpy.maximum(dist_mat_trimmed_wgt, numpy.ones(dist_mat_trimmed_wgt.shape))
    return dist_mat_trimmed, dist_mat_trimmed_wgt

def intrinsic_isomaps(dist_mat_geo, dist_mat_short, dim_intrinsic, noisy_sensor_clusters_2):

    dist_mat_corrected, dist_mat_local_flat_wgt = trim_non_euc(dist_mat_geo, dist_mat_short,  dim_intrinsic, noisy_sensor_clusters_2)
    dist_mat_corrected, dist_mat_local_flat_wgt = trim_non_euc2(dist_mat_corrected,  dim_intrinsic, noisy_sensor_clusters_2)

    D_squared = dist_mat_corrected ** 2

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
    iso_embedding = mds.fit(dist_mat_geo, init=eigen_vect).embedding_
    #print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters_2,
    #              titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
    # stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_local.T, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")
    mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)

    iso_embedding = mds.fit(dist_mat_short, weight=(dist_mat_short != 0).astype(int),
                                  init=iso_embedding).embedding_
    #print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters_2,
    #              titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
    #stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_local.T,
    #                                           titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")
    return iso_embedding

def trim_non_euc2(dist_mat_trust, dim_intrinsic, intrinsic_process_clusters):
    dist_mat_fill = scipy.sparse.csgraph.shortest_path(dist_mat_trust, directed=False)

    n_points = dist_mat_trust.shape[0]
    dist_mat_trimmed = numpy.zeros((n_points, n_points))
    dist_mat_trimmed_wgt = numpy.zeros((n_points, n_points))
    #indexs_balls = numpy.random.choice(n_points, size=n_points, replace=False)

    for i_point in range(1):
        dist_mat_trust_temp = numpy.array(dist_mat_trust, copy=True)

        D_fill = scipy.sparse.csgraph.shortest_path(dist_mat_trust_temp, directed=False)

        knn_indexes = numpy.argsort(dist_mat_fill[i_point], kind='quicksort')
        n_neighbors = 40
        flat = True
        check_list = []
        while flat:
            knn_indexes_sub = knn_indexes[0:n_neighbors]

            D_sub_trust_original = dist_mat_trust[knn_indexes_sub, :][:, knn_indexes_sub]


            numpy.fill_diagonal(D_fill, 0)

            D_fill_sub = D_fill[knn_indexes_sub, :][:, knn_indexes_sub]

            # square it
            D_squared = D_fill_sub ** 2

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

            guess = numpy.real(numpy.dot(numpy.diag(numpy.sqrt(numpy.abs(eigen_val[:dim_intrinsic]))), eigen_vect.T).T)

            wgt = (D_sub_trust_original != 0).astype(int)

            mds = manifold.MDS(n_components=dim_intrinsic, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=1, n_init=1)
            #flat_local = mds.fit(D_fill_sub, init=guess).embedding_
            #stress1 = mds.stress_
            flat_local = mds.fit((D_sub_trust_original+D_sub_trust_original.T)/2, weight=wgt, init=guess).embedding_
            stress2 = mds.stress_

            flat_local = flat_local.T
            #fig = plt.figure()
            #ax = fig.gca()
            #ax.scatter(flat_local[0, :], flat_local[1, :], c="k")

            #expl = numpy.sum(eigen_val[:dim_intrinsic])
            #res = numpy.sum(eigen_val[dim_intrinsic:])
            dis = (D_sub_trust_original*wgt).sum()
            check = (stress2/dis)
            check_list.append(check)
            #flat = (check < 0.05)

            dis = numpy.sqrt(calc_dist(flat_local))
            for i_row in range(knn_indexes_sub.shape[0]):
                for i_col in range(knn_indexes_sub.shape[0]):
                    dist_mat_trust_temp[knn_indexes_sub[i_row], knn_indexes_sub[i_col]] = dis[i_row, i_col]

            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(intrinsic_process_clusters[0, :], intrinsic_process_clusters[1, :], c="k")
            for j_point in knn_indexes_sub:
                ax.scatter(intrinsic_process_clusters[0, j_point], intrinsic_process_clusters[1, j_point], c='r')
            ax.scatter(intrinsic_process_clusters[0, knn_indexes_sub[0]],
                       intrinsic_process_clusters[1, knn_indexes_sub[0]], c='g')
            plt.axis('equal')

            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(flat_local[0, 0], flat_local[1, 0], c="g")
            ax.scatter(flat_local[0, 1:], flat_local[1, 1:], c="r")
            plt.axis('equal')
            plt.show(block=False)

            if n_neighbors == n_points:
                break


            n_neighbors = min(numpy.ceil(n_neighbors*1.5), n_points)


        '''
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(intrinsic_process_clusters[0, :], intrinsic_process_clusters[1, :], c="k")
        for j_point in knn_indexes_sub:
            ax.scatter(intrinsic_process_clusters[0, j_point], intrinsic_process_clusters[1, j_point], c='r')
        ax.scatter(intrinsic_process_clusters[0, knn_indexes_sub[0]], intrinsic_process_clusters[1, knn_indexes_sub[0]], c='g')
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(flat_local[0, 0], flat_local[1, 0], c="g")
        ax.scatter(flat_local[0, 1:], flat_local[1, 1:], c="r")
        plt.axis('equal')
        plt.show(block=False)
        '''

        print(i_point)

        #dist_mat_trimmed = dist_mat_trimmed + dist_mat_trust_temp

        #dist_mat_trimmed_wgt = dist_mat_trimmed_wgt +1
        for i_row in knn_indexes_sub:
            for i_col in knn_indexes_sub:
                dist_mat_trimmed[i_row, i_col] = dist_mat_trimmed[i_row, i_col] + dist_mat_trust_temp[i_row, i_col]
                dist_mat_trimmed_wgt[i_row, i_col] = dist_mat_trimmed_wgt[i_row, i_col] + 1

    dist_mat_trimmed = dist_mat_trimmed/numpy.maximum(dist_mat_trimmed_wgt, numpy.ones(dist_mat_trimmed_wgt.shape))
    return dist_mat_trimmed, dist_mat_trimmed_wgt

def test_ml(Y, X, n_neighbors, n_components, color):

    X = X.T
    Y = Y.T

    fig = plt.figure(figsize=(15, 8))
    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (1000, n_neighbors), fontsize=14)

    try:
        # compatibility matplotlib < 1.0
        ax = fig.add_subplot(251, projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        #ax._axis3don = False
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=color)
        ax.view_init(10, 60)
        plt.title("Observed")
    except:
        ax = fig.add_subplot(251, projection='3d')
        plt.scatter(Y[:, 0], Y[:, 2], c=color)

    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    for i, method in enumerate(methods):
        Z = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='auto',
                                            method=method).fit_transform(Y)

        ax = fig.add_subplot(252 + i)
        plt.scatter(Z[:, 0], Z[:, 1], c=color)
        plt.title("%s" % (labels[i]))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    ax = fig.add_subplot(256)

    plt.scatter(X[:, 0], X[:, 1], c=color)
    plt.title("Intrinsic")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    Z = manifold.Isomap(n_neighbors, n_components).fit_transform(Y)
    ax = fig.add_subplot(257)
    plt.scatter(Z[:, 0], Z[:, 1], c=color)
    plt.title("Isomap")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    Z = mds.fit_transform(Y)
    ax = fig.add_subplot(258)
    plt.scatter(Z[:, 0], Z[:, 1], c=color)
    plt.title("MDS")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    se = manifold.SpectralEmbedding(n_components=n_components,
                                    n_neighbors=n_neighbors)
    Z = se.fit_transform(Y)
    ax = fig.add_subplot(259)
    plt.scatter(Z[:, 0], Z[:, 1], c=color)
    plt.title("SpectralEmbedding")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    plt.axis('tight')

    plt.show()











