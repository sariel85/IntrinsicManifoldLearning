import scipy
from scipy.spatial.distance import cdist
import numpy
import matplotlib.pyplot as plt
from math import sqrt


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
            dist_mat[:, i_x] = (numpy.tile(a2, (n_points, 1)) + b2.T.reshape((n_points, 1)) - 2 * ab.reshape((n_points, 1))).reshape(n_points)

    dist_mat = (dist_mat + dist_mat.T)/2

    return dist_mat

def get_metrics_from_points(cluster_centers, input_base, input_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance):

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
        if s_def[dim_intrinsic:] < numpy.finfo(numpy.float32).eps:
            s_full[dim_intrinsic:] = numpy.finfo(numpy.float32).eps

        s_full = 1 / s_full
        cov_list_def[x] = intrinsic_variance*numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        cov_list_full[x] = intrinsic_variance*numpy.dot(U, numpy.dot(numpy.diag(s_full), V))


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
    return dist_mat_trim


def calc_diff_map(dist_mat, dims=2, factor=2):
    sigma = numpy.median(dist_mat)/factor
    diff_kernal = numpy.exp(-(dist_mat ** 2) / (2 * sigma ** 2))
    row_sum = numpy.sum(diff_kernal, axis=1)
    normlized_kernal = numpy.dot(numpy.diag(1 / row_sum), diff_kernal)
    U, S, V = numpy.linalg.svd(normlized_kernal)
    return U[:, 1:dims+1]

def print_metrics(noisy_sensor_clusters, metric_list, intrinsic_dim, titleStr, scale, space_mode):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    n_points = noisy_sensor_clusters.shape[1]

    if space_mode is True:
        for i_point in range(0, n_points):
            metric = metric_list[i_point]
            [u, s, v] = numpy.linalg.svd(metric)
            u = numpy.dot(u[:, 0:intrinsic_dim], numpy.diag(numpy.sqrt(1/s[:intrinsic_dim])))
            ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                      noisy_sensor_clusters[2, i_point], u[0, 0], u[1, 0], u[2, 0],
                      length=3*numpy.sqrt(scale), pivot='tail')
            ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                      noisy_sensor_clusters[2, i_point], u[0, 1], u[1, 1], u[2, 1],
                      length=3*numpy.sqrt(scale), pivot='tail')
    else:
        for i_point in range(0, n_points):
            metric = metric_list[i_point]
            [u, s, v] = numpy.linalg.svd(metric)
            u = numpy.dot(u[:, 0:intrinsic_dim], numpy.sqrt(scale)*numpy.diag(numpy.sqrt(1/s[:intrinsic_dim])))
            ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                      noisy_sensor_clusters[2, i_point], u[0, 0], u[1, 0], u[2, 0],
                      length=numpy.linalg.norm(u[:, 0]), pivot='tail')
            ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                      noisy_sensor_clusters[2, i_point], u[0, 1], u[1, 1], u[2, 1],
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













