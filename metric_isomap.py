from __future__ import print_function

import numpy


def metric_isomap(noisy_sensor_clusters, metric_list, k neighbors, dim_intrinsic=2):
    n_iters = 5
    n_points = noisy_sensor_clusters.shape[0]
    k_neighboors
    charts = numpy.zeros(shape=(n_points, n_points, dim_intrinsic))
    charts_iter = numpy.zeros(shape=(n_points, n_points, dim_intrinsic))
    for i_iter in range(0, n_iters):
        for i_point_base in range(0, n_points):
            charts_mean_count = numpy.zeros(shape=(n_points))
            for i_neighboors in range(0, k_neighboors):
                for i_point_target in range(0, n_points):
                    if charts[i_neighboors, i_point_target, :] not zeros:
                        new_estimation = noisy_sensor_clusters[i_point_base, i_neighboors,:]
                        charts_iter[i_neighboors, i_point_target, :] = charts_iter[i_neighboors, i_point_target, :]
                        charts_mean_count[i_point_target] = charts_mean_count[i_point_target]+1







